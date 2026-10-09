import React, { useState } from 'react';
import { FolderSync, RefreshCw } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys, useConnectors } from '../api/queries';
import type { Connector } from '../api/types';
import { hasFirmRole, useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';
import NotAvailable from './NotAvailable';

const when = (iso?: string | null) => (iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : 'never');
const RUN: Record<string, [string, Tone]> = { ok: ['Synced', 'verified'], partial: ['Synced, with skipped files', 'attention'], failed: ['Failed', 'conflict'] };
const COUNTS: [string, string][] = [['imported', 'imported'], ['updated', 'updated'], ['unchanged', 'unchanged'], ['removed', 'removed at the source'],
  ['held', 'kept under a legal hold'], ['acl_missing', 'skipped: no permissions file'], ['acl_denied', 'skipped: nobody named has an account'],
  ['acl_changed', 'permissions changed'], ['rejected', 'refused: file type or size'], ['skipped_unsafe', 'skipped: link outside the folder'], ['errors', 'errors']];

function state(c: Connector): [string, Tone] {
  if (c.status === 'disabled') return ['Disabled — its documents are hidden', 'neutral'];
  if (!c.fresh) return ['Out of date — its documents are hidden until the next good sync', 'conflict'];
  return c.last_status === 'partial' ? ['Up to date, some files skipped', 'attention'] : ['Up to date', 'verified'];
}

/** Admin → Connectors: where documents can come from. Administrators see counts, never matters. */
export default function Connectors() {
  const me = useSession();
  const allowed = hasFirmRole(me, 'firm_admin');
  const qc = useQueryClient();
  const data = useConnectors(allowed);
  const [form, setForm] = useState({ name: '', root: '', sync_minutes: 15, stale_after_minutes: 60 });
  const [error, setError] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  if (!allowed) return <NotAvailable />;
  const act = async (id: string, fn: () => Promise<unknown>) => {
    setBusy(id); setError('');
    try { await fn(); } catch (e) { setError(e instanceof ApiError ? e.message : 'That did not work.'); }
    finally { setBusy(null); await qc.invalidateQueries({ queryKey: keys.connectors }); }
  };
  const create = (e: React.FormEvent) => {
    e.preventDefault();
    void act('new', async () => { await api.post('/api/legal/connectors', { kind: 'local_folder', ...form }); setForm({ ...form, name: '', root: '' }); });
  };
  return (
    <>
      <Toolbar title="Connectors" />
      <div className="lw-content" style={{ maxWidth: 1000 }}>
        <p className="lw-meta" style={{ margin: 0 }}>Documents can be brought in from the firm’s document system. A matter’s lead links a folder to the matter. A file comes in only with a permissions file naming who may see it; documents from a connector that has not synced recently are hidden until it has.</p>
        {error && <ErrorNote>{error}</ErrorNote>}
        {data.isPending ? <div className="lw-paper lw-sheet"><Skeleton /></div> : data.isError ? <ErrorNote>The connectors could not be loaded.</ErrorNote> : <>
          <section className="lw-paper lw-sheet" aria-label="Providers" style={{ display: 'grid', gap: 6 }}>
            <h2 className="lw-h2">Available sources</h2>
            {data.data.providers.map((p) => (
              <div key={p.kind} style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <StatusPill tone={p.available ? 'verified' : 'neutral'}>{p.available ? 'Available' : 'Not connected'}</StatusPill>
                <span>{p.name}</span>{p.note && <span className="lw-meta">— {p.note}</span>}
              </div>
            ))}
          </section>
          {data.data.connectors.length === 0 ? <EmptyState title="No connectors yet" /> : data.data.connectors.map((c) => {
            const [label, tone] = state(c);
            const last = c.runs[0];
            return (
              <section key={c.id} className="lw-paper lw-sheet" aria-label={c.name} style={{ display: 'grid', gap: 10 }}>
                <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                  <FolderSync size={20} style={{ color: 'var(--ink-3)', flex: 'none', marginTop: 2 }} />
                  <div style={{ flex: 1 }}>
                    <h2 className="lw-h2">{c.name}</h2>
                    <p className="lw-meta" style={{ margin: '2px 0 0' }}>Folder on this server · {c.links} linked folder{c.links === 1 ? '' : 's'} · {c.documents} document{c.documents === 1 ? '' : 's'} · syncs every {c.sync_minutes} min · hidden after {c.stale_after_minutes} min without a good sync</p>
                  </div>
                  <StatusPill tone={tone}>{label}</StatusPill>
                </div>
                <p className="lw-meta" style={{ margin: 0 }}>Last sync {when(c.last_sync_at)}{last ? ` — ${RUN[last.status][0].toLowerCase()}` : ''}{c.last_error ? ` (${c.last_error.replace(/_/g, ' ')})` : ''}</p>
                {last && <ul className="lw-chips">{COUNTS.filter(([k]) => Number(last.report[k]) > 0).map(([k, t]) => <li key={k}>{String(last.report[k])} {t}</li>)}</ul>}
                <div style={{ display: 'flex', gap: 8 }}>
                  <button type="button" className="lw-btn lw-btn-sm" disabled={busy === c.id || c.status !== 'active'} onClick={() => act(c.id, () => api.post(`/api/legal/connectors/${c.id}/sync`))}>
                    <RefreshCw />{busy === c.id ? 'Syncing…' : 'Sync now'}</button>
                  {c.status === 'active'
                    ? <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => act(c.id, () => api.post(`/api/legal/connectors/${c.id}/disable`))}>Disable</button>
                    : <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => act(c.id, () => api.post(`/api/legal/connectors/${c.id}/enable`))}>Enable</button>}
                </div>
              </section>
            );
          })}
          <form className="lw-paper lw-sheet" aria-label="Add a folder connector" onSubmit={create} style={{ display: 'grid', gap: 10 }}>
            <h2 className="lw-h2">Add a folder connector</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 10, alignItems: 'start' }}>
              <Field label="Name">{(id) => <input id={id} className="lw-input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />}</Field>
              <Field label="Folder on this server" hint="Must be inside a folder the deployment allows (LEGAL_CONNECTOR_LOCAL_ROOTS).">{(id) => <input id={id} className="lw-input lw-mono" value={form.root} onChange={(e) => setForm({ ...form, root: e.target.value })} />}</Field>
              <Field label="Sync every (minutes)">{(id) => <input id={id} type="number" min={1} className="lw-input" value={form.sync_minutes} onChange={(e) => setForm({ ...form, sync_minutes: Number(e.target.value) })} />}</Field>
              <Field label="Hide documents after (minutes without a good sync)">{(id) => <input id={id} type="number" min={1} className="lw-input" value={form.stale_after_minutes} onChange={(e) => setForm({ ...form, stale_after_minutes: Number(e.target.value) })} />}</Field>
            </div>
            <div><button type="submit" className="lw-btn lw-btn-primary" disabled={!form.name.trim() || !form.root.trim() || busy === 'new'}>Add connector</button></div>
          </form>
        </>}
      </div>
    </>
  );
}
