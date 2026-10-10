import React, { useState } from 'react';
import { Link } from 'react-router';
import { CalendarCheck, Check, CheckCheck, Download, PenLine, RotateCcw, X } from 'lucide-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { obligationsPath, useDocuments, useMemberships, useObligations } from '../api/queries';
import type { Obligation, ObligationEvent, ObligationKind } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';

const KIND: Record<ObligationKind, string> = {
  renewal: 'Renewal', termination: 'Termination', change_of_control: 'Change of control', consent: 'Consent', payment: 'Payment',
  notification: 'Notification', post_termination: 'After termination', confidentiality: 'Confidentiality', notice: 'Notice',
};
const STATUS: Record<Obligation['status'], [string, Tone]> = {
  proposed: ['Proposed — not confirmed', 'attention'], confirmed: ['Confirmed', 'verified'], done: ['Done', 'neutral'], dismissed: ['Dismissed', 'neutral'],
};
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
/** "1 December 2028": a legal date reads the same in every browser, and as the backend writes it. */
const day = (iso: string) => { const [y, m, d] = iso.split('-').map(Number); return `${d} ${MONTHS[m - 1]} ${y}`; };
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/** A matter's obligations and deadlines (S3.6, D-050). Read from the documents by rules, each with the
 *  exact words it came from; a date only where the document gives one. Nothing counts as confirmed until
 *  a lawyer confirms it, and every change keeps who made it and why. */
export default function MatterObligations() {
  const matter = useMatter();
  const qc = useQueryClient();
  const obs = useObligations(matter.id);
  const docs = useDocuments(matter.id);
  const [doc, setDoc] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [msg, setMsg] = useState('');
  const canRead = ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open';
  const refresh = () => qc.invalidateQueries({ queryKey: ['matter', matter.id, 'obligations'] });
  const read = async () => {
    setBusy(true); setError(''); setMsg('');
    try {
      const r = await api.post<{ added: number; model: 'used' | 'off' | 'unavailable' }>(`/api/legal/matters/${encodeURIComponent(matter.id)}/documents/${encodeURIComponent(doc)}/obligations/extract`, {});
      setMsg((r.added ? `${r.added} ${r.added === 1 ? 'obligation' : 'obligations'} found and proposed.` : 'Nothing new: every obligation found is already listed.')
        + (r.model === 'unavailable' ? ' The model could not be reached, so only the rules read this document.' : ''));
      await refresh();
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The document could not be read.'); }
    finally { setBusy(false); }
  };
  const list = obs.data?.obligations ?? [];
  const live = list.filter((o) => o.status === 'proposed' || o.status === 'confirmed');
  const overdue = live.filter((o) => o.overdue);
  const dated = live.filter((o) => o.due_date && !o.overdue);
  const undated = live.filter((o) => !o.due_date);
  const closed = list.filter((o) => o.status === 'done' || o.status === 'dismissed');
  const proposed = live.filter((o) => o.status === 'proposed').length;
  const searchable = (docs.data ?? []).filter((d) => d.current_version?.index_status === 'indexed');
  return (
    <>
      <Toolbar crumb={matter.number} title="Obligations">
        {obs.data?.calendar && <a className="lw-btn lw-btn-sm" href={`${obligationsPath(matter.id)}.ics`} download><Download />Calendar file</a>}
      </Toolbar>
      <div className="lw-content">
        {canRead && (
          <section className="lw-paper lw-sheet" aria-label="Read a document's obligations" aria-busy={busy}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
              <div style={{ minWidth: 300 }}>
                <Field label="Document">{(id) => <select id={id} className="lw-input" value={doc} onChange={(e) => setDoc(e.target.value)}>
                  <option value="">Choose a document</option>{searchable.map((d) => <option key={d.id} value={d.id}>{d.title}</option>)}</select>}</Field>
              </div>
              <button type="button" className="lw-btn lw-btn-primary" disabled={!doc || busy} onClick={read}><CalendarCheck />{busy ? 'Reading…' : 'Read its obligations'}</button>
            </div>
            <p className="lw-meta" style={{ margin: '8px 0 0' }}>Terms, notice periods, renewal and break dates, termination and consent rights, payment, notification and other duties, read from the document's own words: by rules first, then by the model, whose every proposal is checked against the sentence. Each is proposed until a lawyer confirms it; a date is given only where the document fixes one.</p>
            {msg && <p className="lw-meta" role="status" style={{ margin: '6px 0 0' }}>{msg}</p>}
            {error && <ErrorNote>{error}</ErrorNote>}
          </section>
        )}
        {obs.isPending ? <div className="lw-paper lw-sheet"><Skeleton /></div>
          : obs.isError ? <ErrorNote>The obligations could not be loaded.</ErrorNote>
            : !list.length ? <EmptyState title="No obligations yet">Choose a contract and read its obligations: each comes with the words it was read from.</EmptyState>
              : (
                <article className="lw-paper lw-sheet" aria-label="Obligations">
                  <p className="lw-meta" role="status" style={{ margin: '0 0 8px' }}>{[`${live.length} open`, proposed && `${proposed} not yet confirmed`, overdue.length && `${overdue.length} overdue`,
                    dated[0]?.due_date && `next due ${day(dated[0].due_date)}`].filter(Boolean).join(' · ')}{` · dates in ${obs.data!.timezone}`}</p>
                  {([['Overdue', overdue], ['Due on a date', dated], ['Running from an event', undated]] as const).map(([title, items]) => items.length > 0 && (
                    <section key={title} aria-label={title}>
                      <h2 className="lw-h3" style={{ margin: '14px 0 2px' }}>{title}</h2>
                      <ul className="lw-check-lines">{items.map((o) => <Item key={o.id} o={o} />)}</ul>
                    </section>
                  ))}
                  {closed.length > 0 && (
                    <details style={{ marginTop: 12 }}><summary className="lw-meta">{`Done or dismissed (${closed.length})`}</summary>
                      <ul className="lw-check-lines">{closed.map((o) => <Item key={o.id} o={o} />)}</ul></details>
                  )}
                </article>
              )}
      </div>
    </>
  );
}

function Item({ o }: { o: Obligation }) {
  const matter = useMatter();
  const qc = useQueryClient();
  const [mode, setMode] = useState<'' | 'confirm' | 'edit' | 'dismiss' | 'history'>('');
  const [error, setError] = useState('');
  const lawyer = ['lead', 'attorney', 'reviewer'].includes(matter.my_role ?? '');
  const drafter = ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open';
  const [label, tone] = STATUS[o.status];
  const title = o.action.charAt(0).toUpperCase() + o.action.slice(1);
  const at = `/m/${matter.number}/documents/${encodeURIComponent(o.document_id)}/read?v=${encodeURIComponent(o.version_id)}&at=${o.evidence.start}-${o.evidence.end}`;
  const send = async (body: Record<string, unknown>) => {
    setError('');
    try {
      await api.patch(`${obligationsPath(matter.id)}/${encodeURIComponent(o.id)}`, { expected_version: o.version, ...body });
      setMode('');
      await qc.invalidateQueries({ queryKey: ['matter', matter.id, 'obligations'] });
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The change could not be saved.'); }
  };
  return (
    <li aria-label={`${title}: ${label}`} data-class={o.overdue ? 'outside' : o.status}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <b className="lw-serif">{title}</b>
        <StatusPill tone="neutral">{KIND[o.kind]}</StatusPill>
        <StatusPill tone={tone}>{o.status === 'confirmed' && o.confirmed_by ? `Confirmed by ${o.confirmed_by}` : label}</StatusPill>
        {o.overdue && <StatusPill tone="conflict">Overdue</StatusPill>}
        <span className="lw-cite-actions">
          <button type="button" className="lw-cite-act lw-meta" aria-expanded={mode === 'history'} onClick={() => setMode(mode === 'history' ? '' : 'history')}>History</button>
          <Link className="lw-cite-act lw-meta" to={at}>Open the words</Link>
        </span>
      </div>
      <p className="lw-meta" style={{ margin: '2px 0' }}>{[o.party, o.clause && `clause ${o.clause}`, o.document_title, o.owner && `owner ${o.owner}`,
        o.source === 'rules' ? 'read by rules' : o.source === 'model' ? 'proposed by the model, its words checked' : `added by ${o.created_by}`].filter(Boolean).join(' · ')}</p>
      {o.when && <p style={{ margin: '2px 0' }}>{`When: ${o.when}`}</p>}
      {o.due_date
        ? <p style={{ margin: '2px 0' }}><b>{`Due ${day(o.due_date)}`}</b><span className="lw-meta">{` (end of day, ${o.timezone})`}</span></p>
        : null}
      {o.due_note && <p className="lw-meta" style={{ margin: '2px 0' }}>{o.due_note}</p>}
      <blockquote className="lw-serif lw-check-quote">{`“${o.evidence.quote}”`}</blockquote>
      {mode === '' && (o.status === 'proposed' || o.status === 'confirmed') && (lawyer || drafter) && (
        <div className="lw-check-decide" role="group" aria-label={`Decide on ${title}`}>
          {lawyer && o.status === 'proposed' && <button type="button" className="lw-btn lw-btn-sm" onClick={() => setMode('confirm')}><Check />Confirm</button>}
          {drafter && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setMode('edit')}><PenLine />Edit</button>}
          {drafter && o.status === 'confirmed' && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => send({ action: 'done' })}><CheckCheck />Mark done</button>}
          {lawyer && o.status === 'proposed' && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setMode('dismiss')}><X />Dismiss</button>}
        </div>
      )}
      {mode === '' && (o.status === 'done' || o.status === 'dismissed') && drafter && (
        <div><button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => send({ action: 'reopen' })}><RotateCcw />Reopen</button></div>
      )}
      {mode === 'confirm' && <ConfirmForm o={o} title={title} onSend={send} onCancel={() => setMode('')} />}
      {mode === 'dismiss' && <ReasonForm label="Why it is not an obligation to track" button="Dismiss" onSend={(reason) => send({ action: 'dismiss', reason })} onCancel={() => setMode('')} />}
      {mode === 'edit' && <EditForm o={o} title={title} onSend={send} onCancel={() => setMode('')} />}
      {mode === 'history' && <History o={o} title={title} />}
      {error && <ErrorNote>{error}</ErrorNote>}
    </li>
  );
}

function ConfirmForm({ o, title, onSend, onCancel }: { o: Obligation; title: string; onSend: (b: Record<string, unknown>) => void; onCancel: () => void }) {
  const matter = useMatter();
  const members = useMemberships(matter.id);
  const [owner, setOwner] = useState(o.owner ?? '');
  return (
    <form className="lw-check-decide" aria-label={`Confirm ${title}`} onSubmit={(e) => { e.preventDefault(); onSend({ action: 'confirm', owner: owner || null }); }}>
      <label className="lw-meta" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>Owner
        <select className="lw-input lw-input-sm" value={owner} onChange={(e) => setOwner(e.target.value)}>
          <option value="">No owner yet</option>{(members.data ?? []).map((m) => <option key={m.id} value={m.username}>{m.display_name || m.username}</option>)}</select></label>
      <button type="submit" className="lw-btn lw-btn-sm lw-btn-primary"><Check />Confirm this obligation</button>
      <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={onCancel}>Cancel</button>
    </form>
  );
}

function ReasonForm({ label, button, onSend, onCancel }: { label: string; button: string; onSend: (reason: string) => void; onCancel: () => void }) {
  const [reason, setReason] = useState('');
  return (
    <form className="lw-check-decide" aria-label={label} onSubmit={(e) => { e.preventDefault(); onSend(reason.trim()); }}>
      <input aria-label={label} className="lw-input lw-input-sm" placeholder={label} value={reason} onChange={(e) => setReason(e.target.value)} />
      <button type="submit" className="lw-btn lw-btn-sm" disabled={reason.trim().length < 3}>{button}</button>
      <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={onCancel}>Cancel</button>
    </form>
  );
}

function EditForm({ o, title, onSend, onCancel }: { o: Obligation; title: string; onSend: (b: Record<string, unknown>) => void; onCancel: () => void }) {
  const [due, setDue] = useState(o.due_date ?? '');
  const [tz, setTz] = useState(o.timezone);
  const [party, setParty] = useState(o.party ?? '');
  const [whenText, setWhenText] = useState(o.when ?? '');
  const [reason, setReason] = useState('');
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const body: Record<string, unknown> = { action: 'edit', reason: reason.trim() };
    if (due && due !== o.due_date) body.due_date = due;
    if (tz !== o.timezone) body.timezone = tz;
    if (party !== (o.party ?? '')) body.party = party;
    if (whenText !== (o.when ?? '')) body.when = whenText;
    onSend(body);
  };
  return (
    <form onSubmit={submit} aria-label={`Edit ${title}`} className="lw-minutes-edit" style={{ display: 'grid', gap: 8, marginTop: 6 }}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <label className="lw-label">Due date<input type="date" className="lw-input lw-input-sm" value={due} onChange={(e) => setDue(e.target.value)} /></label>
        <label className="lw-label">Timezone<input className="lw-input lw-input-sm" value={tz} onChange={(e) => setTz(e.target.value)} placeholder="Europe/London" /></label>
        <label className="lw-label">Party<input className="lw-input lw-input-sm" value={party} onChange={(e) => setParty(e.target.value)} /></label>
      </div>
      <label className="lw-label">When<input className="lw-input lw-input-sm" value={whenText} onChange={(e) => setWhenText(e.target.value)} /></label>
      <label className="lw-label">Why (kept with the change)<input className="lw-input lw-input-sm" value={reason} onChange={(e) => setReason(e.target.value)} /></label>
      <div style={{ display: 'flex', gap: 6 }}>
        <button type="submit" className="lw-btn lw-btn-sm lw-btn-primary" disabled={reason.trim().length < 3}>Save</button>
        <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

function History({ o, title }: { o: Obligation; title: string }) {
  const matter = useMatter();
  const q = useQuery({ queryKey: ['matter', matter.id, 'obligation', o.id, o.version], queryFn: () => api.get<{ history: ObligationEvent[] }>(`${obligationsPath(matter.id)}/${encodeURIComponent(o.id)}`) });
  if (q.isPending) return <Skeleton />;
  if (q.isError) return <ErrorNote>The history could not be loaded.</ErrorNote>;
  const said: Record<string, string> = { proposed: 'proposed', confirm: 'confirmed', edit: 'edited', done: 'marked done', dismiss: 'dismissed', reopen: 'reopened' };
  return (
    <ol className="lw-meta" aria-label={`History of ${title}`} style={{ margin: '4px 0', paddingLeft: 18 }}>
      {q.data.history.map((h, i) => (
        <li key={i}>{`${h.by ?? 'someone'} ${said[h.action] ?? h.action}, ${when(h.at)}`}
          {Object.entries(h.changes).filter(([k]) => k !== 'status').map(([k, [a, b]]) => ` · ${k.replace('_', ' ')}: ${a ?? '—'} → ${b ?? '—'}`).join('')}
          {h.reason ? ` — ${h.reason}` : ''}</li>
      ))}
    </ol>
  );
}
