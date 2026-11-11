import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { Columns3, FileSpreadsheet, Plus } from 'lucide-react';
import { api, ApiError } from '../api/client';
import { keys, useMatters, useTracks } from '../api/queries';
import type { ImportRow } from '../api/types';
import { hasFirmRole, useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { trackOf } from '../ui/casework';
import { Dialog, EmptyState, ErrorNote, MatterStatusPill, roleLabel, SensitivityPill, Skeleton } from '../ui/primitives';

export default function Matters() {
  const me = useSession();
  const matters = useMatters();
  const tracks = useTracks();
  const navigate = useNavigate();
  const [importing, setImporting] = useState(false);
  const canCreate = hasFirmRole(me, 'attorney', 'managing_partner');
  const stageOf = (m: { track?: string | null; stage?: string | null }) =>
    trackOf(tracks.data, m as never)?.stages.find((s) => s.key === m.stage)?.label ?? '';
  return (
    <>
      <Toolbar title="Matters">
        <Link to="/pipeline" className="lw-btn lw-btn-ghost"><Columns3 />Pipeline</Link>
        {canCreate && <button type="button" className="lw-btn" onClick={() => setImporting(true)}><FileSpreadsheet />Import cases</button>}
        {canCreate && <Link to="/matters/new" className="lw-btn lw-btn-primary"><Plus />New matter</Link>}
      </Toolbar>
      <div className="lw-content">
        {matters.isPending ? <div className="lw-paper lw-sheet" style={{ display: 'grid', gap: 10 }}><Skeleton /><Skeleton /><Skeleton /></div>
          : matters.error ? <EmptyState title="Matters could not be loaded">Check the connection and try again.</EmptyState>
            : matters.data.length === 0 ? (
              <EmptyState title="No matters yet" action={canCreate ? <button type="button" className="lw-btn lw-btn-primary" onClick={() => navigate('/matters/new')}><Plus />New matter</button> : undefined}>
                {canCreate ? 'Open a case to start working with its documents, updates, plan and meetings.' : 'Matters you are a member of will appear here. The lead on a matter can add you.'}
              </EmptyState>
            ) : (
              <div className="lw-paper" style={{ overflowX: 'auto' }}>
                <table className="lw-table">
                  <thead><tr><th>Matter</th><th>Client</th><th>Stage</th><th>Status</th><th>Sensitivity</th><th>Your role</th></tr></thead>
                  <tbody>
                    {matters.data.map((m) => (
                      <tr key={m.id}>
                        <td><Link to={`/m/${encodeURIComponent(m.number)}/overview`} style={{ color: 'inherit', textDecoration: 'none' }}>
                          <div className="lw-tnum" style={{ fontWeight: 600, color: 'var(--accent)' }}>{m.number}</div><div>{m.title}</div></Link></td>
                        <td>{m.client.name}</td>
                        <td>{stageOf(m)}</td>
                        <td><MatterStatusPill value={m.status} /></td>
                        <td><SensitivityPill value={m.sensitivity} /></td>
                        <td>{roleLabel(m.my_role)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
      </div>
      {importing && <ImportDialog onClose={() => setImporting(false)} />}
    </>
  );
}

/** Opening many matters from a spreadsheet saved as CSV: every row is checked and shown first; only rows without a problem
 *  are opened, each with the importer as its lead. */
function ImportDialog({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [rows, setRows] = useState<ImportRow[] | null>(null);
  const [created, setCreated] = useState<{ row: number; number: string }[] | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const send = async (f: File, preview: boolean) => {
    setError(''); setBusy(true);
    try {
      const body = new FormData();
      body.append('file', f);
      const r = await api.upload<{ rows: ImportRow[]; ready: number; created: { row: number; number: string }[] }>(`/api/legal/matters-import?preview=${preview}`, body);
      setRows(r.rows);
      if (!preview) {
        setCreated(r.created);
        await Promise.all([qc.invalidateQueries({ queryKey: keys.matters }), qc.invalidateQueries({ queryKey: keys.clients }),
                           qc.invalidateQueries({ queryKey: keys.pipeline })]);
      }
    } catch (e) { setError(e instanceof ApiError ? e.message : 'The file could not be read.'); }
    finally { setBusy(false); }
  };
  const ready = rows?.filter((r) => !r.errors.length).length ?? 0;
  return (
    <Dialog title="Import cases" onClose={onClose} width={860}
      footer={<>
        <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>{created ? 'Done' : 'Cancel'}</button>
        {!created && rows && <button type="button" className="lw-btn lw-btn-primary" disabled={!ready || busy || !file} onClick={() => file && void send(file, false)}>
          {busy ? 'Opening…' : `Open ${ready} matter${ready === 1 ? '' : 's'}`}</button>}
      </>}>
      <div style={{ display: 'grid', gap: 12 }}>
        <p style={{ margin: 0 }}>Save the spreadsheet as CSV with a header row. Needed: <b>number</b>, <b>title</b>, <b>client</b>. Optional: client_code, practice_area, jurisdiction,
          sensitivity, track, stage, description, other_parties (separated by “;”). Up to 200 rows. You become the lead of each matter.</p>
        <p style={{ margin: 0 }}><a className="qc-linkbtn" href="/api/legal/matters-import/template" download>Download the template</a></p>
        {!created && (
          <label className="lw-btn" style={{ justifySelf: 'start' }}>
            <FileSpreadsheet />{file ? `Choose another file (${file.name})` : 'Choose a CSV file'}
            <input type="file" accept=".csv,text/csv" hidden onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; if (f) { setFile(f); setCreated(null); void send(f, true); } }} />
          </label>
        )}
        {error && <ErrorNote>{error}</ErrorNote>}
        {created && <p role="status" style={{ margin: 0 }}><b>{created.length} matter{created.length === 1 ? '' : 's'} opened:</b> {created.map((c) => c.number).join(', ')}.</p>}
        {rows && (
          <div style={{ overflowX: 'auto', maxHeight: 360 }}>
            <table className="lw-table" aria-label="Rows in the file">
              <thead><tr><th>Row</th><th>Matter</th><th>Client</th><th>Track and stage</th><th>Check</th></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.row} data-problem={r.errors.length ? 'true' : undefined}>
                    <td className="lw-tnum">{r.row}</td>
                    <td><b className="lw-tnum">{r.number || '—'}</b><div>{r.title || '—'}</div></td>
                    <td>{r.client || '—'}{r.new_client && <div className="lw-meta">New client</div>}</td>
                    <td>{r.track ? `${r.track[0].toUpperCase()}${r.track.slice(1)} · ${r.stage_label}` : '—'}</td>
                    <td>{r.errors.length ? <span style={{ color: 'var(--conflict, #b3261e)' }}>{r.errors.join(' ')}</span>
                      : created?.some((c) => c.row === r.row) ? 'Opened' : 'Ready'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Dialog>
  );
}
