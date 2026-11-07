/**
 * Opening a new case (case management, 2026-11-07), in three steps: the details (client, number, title, track and stage),
 * the people (the team and the other side, so that conflict checks find it), and the documents (several files, a folder
 * or a zip; this step can be skipped). The person opening it becomes its lead; a colleague screened from the client by an
 * ethical wall cannot be added.
 */
import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { ArrowLeft, ArrowRight, Plus, Trash2 } from 'lucide-react';
import { api, ApiError } from '../api/client';
import { keys, useClients, useColleagues, useTracks } from '../api/queries';
import type { Client, Matter, MatterRole, Sensitivity, TrackKey } from '../api/types';
import { hasFirmRole, useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import BulkUpload from '../ui/bulkUpload';
import { EmptyState, ErrorNote, Field, roleLabel } from '../ui/primitives';

const ROLES: MatterRole[] = ['attorney', 'paralegal', 'reviewer', 'read_only', 'lead'];
const PARTY_ROLES: [string, string][] = [['adverse', 'Other side'], ['related', 'Related party'], ['client_group', "Client's group"], ['other', 'Other']];
const suggestTrack = (area: string): TrackKey => {
  const a = area.toLowerCase();
  if (/litigat|disput|employment|arbitrat|insolvenc/.test(a)) return 'disputes';
  if (/corporate|m&a|commercial|real estate|property|finance|transaction/.test(a)) return 'transactions';
  return 'advisory';
};

export default function NewMatter() {
  const me = useSession();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const clients = useClients();
  const tracks = useTracks();
  const opener = hasFirmRole(me, 'attorney', 'managing_partner');
  const colleagues = useColleagues(opener);
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [clientId, setClientId] = useState('');
  const [newClient, setNewClient] = useState({ name: '', code: '' });
  const [form, setForm] = useState({ number: '', title: '', practice_area: '', jurisdiction: '', sensitivity: 'restricted' as Sensitivity,
                                     description: '', track: '' as TrackKey | '', stage: '' });
  const [team, setTeam] = useState<{ username: string; role: MatterRole }[]>([]);
  const [parties, setParties] = useState<{ name: string; role: string }[]>([]);
  const [party, setParty] = useState({ name: '', role: 'adverse' });
  const [created, setCreated] = useState<Matter | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!opener) return;
    api.get<{ number: string }>('/api/legal/matter-numbers/next?prefix=M-').then((r) => setForm((f) => (f.number ? f : { ...f, number: r.number }))).catch(() => undefined);
  }, [opener]);
  if (!opener) {
    return <div className="lw-content"><EmptyState title="Opening matters is for attorneys and managing partners">Ask one of them to open the matter and add you to it.</EmptyState></div>;
  }
  const track = (form.track || suggestTrack(form.practice_area)) as TrackKey;
  const trackDef = tracks.data?.find((t) => t.key === track);
  const stage = form.stage && trackDef?.stages.some((s) => s.key === form.stage) ? form.stage : trackDef?.stages[0]?.key ?? '';
  const detailsReady = form.number.trim() && form.title.trim() && clientId && (clientId !== '__new' || newClient.name.trim());
  const create = async () => {
    setError(''); setBusy(true);
    try {
      let cid = clientId;
      if (cid === '__new') cid = (await api.post<{ client: Client }>('/api/legal/clients', { name: newClient.name.trim(), code: newClient.code.trim() || undefined })).client.id;
      const m = (await api.post<{ matter: Matter }>('/api/legal/matters', {
        client_id: cid, number: form.number.trim(), title: form.title.trim(), sensitivity: form.sensitivity,
        practice_area: form.practice_area.trim() || undefined, jurisdiction: form.jurisdiction.trim() || undefined,
        description: form.description.trim() || undefined, track, stage: stage || undefined, members: team, parties,
      })).matter;
      await Promise.all([qc.invalidateQueries({ queryKey: keys.matters }), qc.invalidateQueries({ queryKey: keys.clients }),
                         qc.invalidateQueries({ queryKey: keys.pipeline })]);
      setCreated(m);
      setStep(3);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The matter could not be opened.');
      if (clientId === '__new') await qc.invalidateQueries({ queryKey: keys.clients });
    } finally { setBusy(false); }
  };
  const open = () => created && navigate(`/m/${encodeURIComponent(created.number)}/overview`);
  const people = (colleagues.data ?? []).filter((c) => !team.some((t) => t.username === c.username));
  return (
    <>
      <Toolbar title="New case" crumb="Matters">
        <Link to="/matters" className="lw-btn lw-btn-ghost"><ArrowLeft />All matters</Link>
      </Toolbar>
      <div className="lw-content" style={{ maxWidth: 860 }}>
        <ol className="lw-steps" aria-label="Steps">
          {['Details', 'People', 'Documents'].map((s, i) => (
            <li key={s} data-state={i + 1 < step ? 'done' : i + 1 === step ? 'current' : 'next'} aria-current={i + 1 === step ? 'step' : undefined}>
              <span aria-hidden>{i + 1}</span>{s}
            </li>
          ))}
        </ol>
        {step === 1 && (
          <form className="lw-paper lw-sheet" style={{ display: 'grid', gap: 12 }} onSubmit={(e) => { e.preventDefault(); if (detailsReady) setStep(2); }}>
            <h2 className="lw-h2" style={{ margin: 0 }}>The case</h2>
            <Field label="Client">{(id) => (
              <select id={id} className="lw-input" value={clientId} onChange={(e) => setClientId(e.target.value)}>
                <option value="" disabled>Choose a client</option>
                {(clients.data ?? []).map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                <option value="__new">New client…</option>
              </select>)}</Field>
            {clientId === '__new' && (
              <div className="lw-grid-2-1">
                <Field label="Client name">{(id) => <input id={id} className="lw-input" maxLength={200} value={newClient.name} onChange={(e) => setNewClient({ ...newClient, name: e.target.value })} />}</Field>
                <Field label="Code" hint="Optional">{(id) => <input id={id} className="lw-input" maxLength={32} value={newClient.code} onChange={(e) => setNewClient({ ...newClient, code: e.target.value })} />}</Field>
              </div>
            )}
            <div className="lw-grid-1-2">
              <Field label="Matter number" hint="The next free number is suggested.">{(id) => <input id={id} className="lw-input lw-tnum" maxLength={64} value={form.number} onChange={(e) => setForm({ ...form, number: e.target.value })} />}</Field>
              <Field label="Title">{(id) => <input id={id} className="lw-input" maxLength={300} value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} placeholder="For example: Late delivery claim against the supplier" />}</Field>
            </div>
            <div className="lw-grid-2">
              <Field label="Practice area" hint="Optional">{(id) => <input id={id} className="lw-input" maxLength={120} value={form.practice_area} onChange={(e) => setForm({ ...form, practice_area: e.target.value })} placeholder="Commercial litigation" />}</Field>
              <Field label="Jurisdiction" hint="Optional">{(id) => <input id={id} className="lw-input" maxLength={120} value={form.jurisdiction} onChange={(e) => setForm({ ...form, jurisdiction: e.target.value })} />}</Field>
            </div>
            <div className="lw-grid-2">
              <Field label="Track" hint={form.track ? undefined : 'Suggested from the practice area.'}>{(id) => (
                <select id={id} className="lw-input" value={track} onChange={(e) => setForm({ ...form, track: e.target.value as TrackKey, stage: '' })}>
                  {(tracks.data ?? []).map((t) => <option key={t.key} value={t.key}>{t.label}</option>)}
                </select>)}</Field>
              <Field label="Starting stage">{(id) => (
                <select id={id} className="lw-input" value={stage} onChange={(e) => setForm({ ...form, stage: e.target.value })}>
                  {(trackDef?.stages ?? []).map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
                </select>)}</Field>
            </div>
            <Field label="What the case is about" hint="Optional. A line or two for the team.">{(id) => (
              <textarea id={id} className="lw-input" rows={3} maxLength={2000} value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />)}</Field>
            <Field label="Sensitivity" hint="Only the matter's members can see it, whatever its sensitivity.">{(id) => (
              <select id={id} className="lw-input" value={form.sensitivity} onChange={(e) => setForm({ ...form, sensitivity: e.target.value as Sensitivity })}>
                <option value="standard">Standard</option><option value="restricted">Restricted</option><option value="highly_restricted">Highly restricted</option>
              </select>)}</Field>
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button type="submit" className="lw-btn lw-btn-primary" disabled={!detailsReady}>Next: people<ArrowRight /></button>
            </div>
          </form>
        )}
        {step === 2 && (
          <section className="lw-paper lw-sheet" style={{ display: 'grid', gap: 16 }}>
            <div>
              <h2 className="lw-h2" style={{ margin: 0 }}>The team</h2>
              <p className="lw-meta" style={{ margin: '2px 0 0' }}>You will be its lead. Add the colleagues who work on it; you can change this later on the Members page.</p>
            </div>
            {team.length > 0 && (
              <ul className="lw-check-lines" aria-label="Team">
                {team.map((t, i) => (
                  <li key={t.username} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <b style={{ flex: 1 }}>{colleagues.data?.find((c) => c.username === t.username)?.display_name ?? t.username}</b>
                    <select className="lw-input" aria-label={`Role of ${t.username}`} style={{ width: 'auto' }} value={t.role}
                      onChange={(e) => setTeam(team.map((x, j) => (j === i ? { ...x, role: e.target.value as MatterRole } : x)))}>
                      {ROLES.map((r) => <option key={r} value={r}>{roleLabel(r)}</option>)}
                    </select>
                    <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" aria-label={`Remove ${t.username}`} onClick={() => setTeam(team.filter((_, j) => j !== i))}><Trash2 /></button>
                  </li>
                ))}
              </ul>
            )}
            <Field label="Add a colleague">{(id) => (
              <select id={id} className="lw-input" value="" onChange={(e) => {
                const c = people.find((x) => x.username === e.target.value);
                if (c) setTeam([...team, { username: c.username, role: c.lawyer ? 'attorney' : 'read_only' }]);
              }}>
                <option value="" disabled>{colleagues.isPending ? 'Loading…' : people.length ? 'Choose a colleague' : 'Everyone is added'}</option>
                {people.map((c) => <option key={c.username} value={c.username}>{c.display_name} ({c.username})</option>)}
              </select>)}</Field>
            <div>
              <h2 className="lw-h2" style={{ margin: 0 }}>The other side</h2>
              <p className="lw-meta" style={{ margin: '2px 0 0' }}>New enquiries are checked against these names, so a conflict is found.</p>
            </div>
            {parties.length > 0 && (
              <ul className="lw-check-lines" aria-label="Parties">
                {parties.map((p, i) => (
                  <li key={`${p.name}-${i}`} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <b style={{ flex: 1 }}>{p.name}</b><span className="lw-meta">{PARTY_ROLES.find(([k]) => k === p.role)?.[1]}</span>
                    <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" aria-label={`Remove ${p.name}`} onClick={() => setParties(parties.filter((_, j) => j !== i))}><Trash2 /></button>
                  </li>
                ))}
              </ul>
            )}
            <form style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}
              onSubmit={(e) => { e.preventDefault(); if (party.name.trim()) { setParties([...parties, { name: party.name.trim(), role: party.role }]); setParty({ ...party, name: '' }); } }}>
              <Field label="Party name">{(id) => <input id={id} className="lw-input" maxLength={160} value={party.name} onChange={(e) => setParty({ ...party, name: e.target.value })} />}</Field>
              <Field label="As">{(id) => (
                <select id={id} className="lw-input" value={party.role} onChange={(e) => setParty({ ...party, role: e.target.value })}>
                  {PARTY_ROLES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
                </select>)}</Field>
              <button type="submit" className="lw-btn" disabled={!party.name.trim() || parties.length >= 20}><Plus />Add party</button>
            </form>
            {error && <ErrorNote>{error}</ErrorNote>}
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
              <button type="button" className="lw-btn lw-btn-ghost" onClick={() => setStep(1)}><ArrowLeft />Back</button>
              <button type="button" className="lw-btn lw-btn-primary" disabled={busy} onClick={() => void create()}>{busy ? 'Opening…' : 'Open the matter'}<ArrowRight /></button>
            </div>
          </section>
        )}
        {step === 3 && created && (
          <section className="lw-paper lw-sheet" style={{ display: 'grid', gap: 14 }}>
            <div>
              <h2 className="lw-h2" style={{ margin: 0 }}>{created.number} is open</h2>
              <p className="lw-meta" style={{ margin: '2px 0 0' }}>Add its documents now, or later from the Documents page. Each one is checked, stored encrypted and searchable only within this matter.</p>
            </div>
            <BulkUpload matterId={created.id} />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button type="button" className="lw-btn lw-btn-primary" onClick={open}>Open the case<ArrowRight /></button>
            </div>
          </section>
        )}
      </div>
    </>
  );
}
