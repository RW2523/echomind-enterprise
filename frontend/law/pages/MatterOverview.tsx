import React, { useState } from 'react';
import { Link } from 'react-router';
import { api, ApiError } from '../api/client';
import { keys, partiesPath, useApiMutation, useParties } from '../api/queries';
import type { Matter, MatterParty, PartyRole, Sensitivity } from '../api/types';
import { useMatter } from '../app/matter';
import { hueOf, initials, matterEntries } from '../app/nav';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { Dialog, EmptyState, ErrorNote, Field, MatterStatusPill, roleLabel, SensitivityPill } from '../ui/primitives';
import { useQueryClient } from '@tanstack/react-query';
import { HoldNotice } from '../ui/records';

export default function MatterOverview() {
  const matter = useMatter();
  const me = useSession();
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
      <div className="lw-content" style={{ maxWidth: 1120 }}>
        <section className="lw-paper lw-sheet" aria-labelledby="matter-title" style={{ display: 'grid', gap: 18 }}>
          <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
            <span className="lw-avatar" style={{ ['--h' as string]: hueOf(matter.client.name), width: 48, height: 48, fontSize: 15, borderRadius: 15 }} aria-hidden>
              {initials(matter.client.name)}
            </span>
            <div>
              <div className="lw-eyebrow">{matter.client.name}</div>
              <h2 className="lw-h1 lw-serif" id="matter-title" style={{ marginTop: 4 }}>{matter.title}</h2>
            </div>
          </div>
          {matter.status !== 'open' && <p className="lw-meta" style={{ margin: 0 }}>This matter is {matter.status}. Its content can be read but not changed until it is reopened.</p>}
          {matter.on_hold && <HoldNotice />}
          {error && <ErrorNote>{error}</ErrorNote>}
          <dl style={{ display: 'grid', gridTemplateColumns: 'max-content 1fr', gap: '10px 24px', margin: 0 }}>
            {rows.map(([k, v]) => (<React.Fragment key={k}><dt className="lw-meta" style={{ fontSize: 13 }}>{k}</dt><dd style={{ margin: 0 }}>{v}</dd></React.Fragment>))}
          </dl>
        </section>
        <section aria-labelledby="matter-work" style={{ display: 'grid', gap: 10 }}>
          <h2 className="lw-h2" id="matter-work" style={{ padding: '0 4px' }}>Work on this matter</h2>
          <div className="lw-tiles">
            {matterEntries(me, matter).filter((e) => e.key !== 'overview').map((e) => (
              <Link key={e.key} to={e.to} className="lw-tile">
                <span className="lw-icon-badge" style={{ ['--h' as string]: e.hue ?? 228 }}><e.icon aria-hidden /></span>
                <span><strong>{e.label}</strong><span className="lw-meta">{e.about}</span></span>
              </Link>
            ))}
          </div>
        </section>
        <Parties matterId={matter.id} />
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

const PARTY_ROLE: Record<PartyRole, string> = { adverse: 'Adverse party', related: 'Related party', client_group: "Client's group", other: 'Other' };

/** The matter's party register (S4.1, D-054): the other side and related parties, which every new enquiry is
 *  screened against. Kept by the matter's lead and attorneys; removing one keeps the history. */
function Parties({ matterId }: { matterId: string }) {
  const q = useParties(matterId);
  const qc = useQueryClient();
  const [name, setName] = useState('');
  const [role, setRole] = useState<PartyRole>('adverse');
  const [removing, setRemoving] = useState<MatterParty | null>(null);
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');
  const refresh = () => qc.invalidateQueries({ queryKey: ['matter', matterId, 'parties'] });
  const add = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try { await api.post(partiesPath(matterId), { name, role }); setName(''); await refresh(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The party could not be added.'); }
  };
  const remove = async () => {
    if (!removing) return;
    setError('');
    try { await api.post(`${partiesPath(matterId)}/${encodeURIComponent(removing.id)}/remove`, { reason }); setRemoving(null); setReason(''); await refresh(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The party could not be removed.'); }
  };
  if (q.isPending || q.isError) return null;
  const { parties, can_manage } = q.data;
  return (
    <section className="lw-paper lw-sheet" aria-labelledby="parties-title" style={{ display: 'grid', gap: 12 }}>
      <div>
        <h2 className="lw-h2" id="parties-title" style={{ margin: 0 }}>Parties</h2>
        <p className="lw-meta" style={{ margin: '2px 0 0' }}>The other side and related parties. New enquiries are checked against these names and the firm's clients.</p>
      </div>
      {parties.length === 0 ? <EmptyState title="No parties recorded">{can_manage ? 'Add the other side so that conflict checks can find this matter.' : 'The matter\'s lawyers add them.'}</EmptyState> : (
        <ul className="lw-check-lines" aria-label="Parties">
          {parties.map((p) => (
            <li key={p.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
              <span><b>{p.name}</b> <span className="lw-meta">{`${PARTY_ROLE[p.role]}${p.added_by ? ` · added by ${p.added_by}` : ''}`}</span></span>
              {can_manage && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setRemoving(p)} aria-label={`Remove ${p.name}`}>Remove</button>}
            </li>
          ))}
        </ul>
      )}
      {can_manage && (
        <form onSubmit={add} style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
          <Field label="Party name">{(id) => <input id={id} className="lw-input" maxLength={160} value={name} onChange={(e) => setName(e.target.value)} />}</Field>
          <Field label="As">{(id) => (
            <select id={id} className="lw-input" value={role} onChange={(e) => setRole(e.target.value as PartyRole)}>
              {(Object.keys(PARTY_ROLE) as PartyRole[]).map((r) => <option key={r} value={r}>{PARTY_ROLE[r]}</option>)}
            </select>)}</Field>
          <button type="submit" className="lw-btn" disabled={!name.trim()}>Add party</button>
        </form>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
      {removing && (
        <Dialog title={`Remove ${removing.name}?`} onClose={() => setRemoving(null)}
          footer={<><button type="button" className="lw-btn" onClick={() => setRemoving(null)}>Cancel</button>
            <button type="button" className="lw-btn lw-btn-primary" disabled={!reason.trim()} onClick={remove}>Remove</button></>}>
          <p style={{ margin: '0 0 10px' }}>Conflict checks will no longer find this matter by this name. The register keeps who added and removed it, and why.</p>
          <Field label="Reason">{(id) => <input id={id} className="lw-input" maxLength={300} value={reason} onChange={(e) => setReason(e.target.value)} />}</Field>
        </Dialog>
      )}
    </section>
  );
}
