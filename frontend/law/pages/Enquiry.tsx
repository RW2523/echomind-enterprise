import React, { useState } from 'react';
import { Check, FolderPlus, Mic, ShieldCheck, ShieldX, Trash2, X } from 'lucide-react';
import { Link, useParams } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { intakePath, keys, useIntake, useIntakes } from '../api/queries';
import type { Intake, IntakeAnswers, IntakeCan, IntakeMatch, IntakeReply, IntakeTemplate, IntakeView } from '../api/types';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { CopyField, Dialog, ErrorNote, Field, Skeleton, StatusPill } from '../ui/primitives';
import { AnswersList, IntakeStatusPill, ScreeningPill, ScriptForm } from '../ui/intake';

const CHANNEL = { phone: 'Telephone', in_person: 'In person', link: 'Online link' } as const;
const TEMPLATE: Record<IntakeTemplate, string> = {
  acknowledge: 'Acknowledge the enquiry', can_help: 'We can help (needs clearance)', cannot_act: 'We cannot help', more_info: 'Ask for more information',
};
const KIND: Record<IntakeMatch['kind'], string> = { client: 'Client', adverse: 'Adverse party', related: 'Related party', client_group: "Client's group", other: 'Other party' };
const ACTION: Record<string, string> = {
  created: 'Enquiry started', invited: 'Link created', disclaimer_accepted: 'Disclaimer accepted', submitted: 'Answers submitted',
  screened: 'Names screened', cleared: 'Cleared by the conflicts team', conflict: 'Conflict confirmed', consent: 'Recording consent given',
  consent_withdrawn: 'Recording consent withdrawn', recording_added: 'Recording attached', recording_removed: 'Recording removed',
  reply_drafted: 'Reply drafted', reply_released: 'Reply approved and released', converted: 'Matter opened', closed: 'Enquiry closed', purged: 'Personal data purged',
};
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
const message = (e: unknown, fallback: string) => (e instanceof ApiError ? e.message : fallback);

/** One enquiry (S4.1, D-054): the script, the screening as staff may see it, the conflicts team's matches and
 *  decision, a call recording under consent, replies a lawyer approves, and opening the matter. */
export default function Enquiry() {
  const { intakeId = '' } = useParams();
  const q = useIntake(intakeId);
  const list = useIntakes();
  const qc = useQueryClient();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const refresh = async () => { await qc.invalidateQueries({ queryKey: ['intake', intakeId] }); await qc.invalidateQueries({ queryKey: ['intakes'] }); };
  const call = async (fn: () => Promise<unknown>, fallback: string) => {
    setError(''); setBusy(true);
    try { await fn(); await refresh(); return true; } catch (e) { setError(message(e, fallback)); return false; } finally { setBusy(false); }
  };
  if (q.isPending || list.isPending) return <div className="lw-content"><Skeleton height={48} /><Skeleton height={240} /></div>;
  if (q.isError || list.isError) return <div className="lw-content"><ErrorNote>This enquiry could not be loaded.</ErrorNote></div>;
  const { intake: i, matches } = q.data;
  const can = list.data.can;
  const base = intakePath(i.id);
  const staffChannel = i.channel !== 'link';
  return (
    <>
      <Toolbar crumb="Enquiries" title={i.reference}>
        <IntakeStatusPill status={i.status} />
      </Toolbar>
      <div className="lw-content" style={{ maxWidth: 960 }}>
        <p className="lw-meta" style={{ margin: 0 }}><Link to="/enquiries">All enquiries</Link>{` · ${CHANNEL[i.channel]} · started ${when(i.created_at)}${i.created_by ? ` by ${i.created_by}` : ''}`}</p>
        {error && <ErrorNote>{error}</ErrorNote>}

        {i.status === 'purged' ? (
          <section className="lw-paper lw-sheet"><p style={{ margin: 0 }}>This enquiry's personal data was purged. The record of what was done, by whom and when, remains below.</p></section>
        ) : (i.status === 'open' || i.status === 'invited') ? (
          <section className="lw-paper lw-sheet" aria-label="Taking the details">
            {i.status === 'invited'
              ? <p style={{ margin: 0 }}>{`The enquirer has a link to answer the questions themselves. It expires ${i.link?.expires_at ? when(i.link.expires_at) : ''}.`}</p>
              : !i.disclaimer.accepted_at ? (
                <div style={{ display: 'grid', gap: 10 }}>
                  <h2 className="lw-h2" style={{ margin: 0 }}>First, the disclaimer</h2>
                  <blockquote className="lw-intake-disclaimer">{i.disclaimer.text}</blockquote>
                  <p className="lw-meta" style={{ margin: 0 }}>Read it to the enquirer. Go on only once they accept it.</p>
                  {can.manage && <div><button type="button" className="lw-btn lw-btn-primary" disabled={busy}
                    onClick={() => call(() => api.post(`${base}/disclaimer`, { version: i.disclaimer.version }), 'The acceptance could not be recorded.')}><Check />The enquirer accepts it</button></div>}
                </div>
              ) : can.manage ? (
                <>
                  <h2 className="lw-h2" style={{ marginTop: 0 }}>The questions</h2>
                  <ScriptForm script={i.script} busy={busy}
                    onSubmit={(answers: IntakeAnswers) => call(() => api.post(`${base}/submit`, { answers }), 'The answers could not be saved.')} />
                </>
              ) : <p style={{ margin: 0 }}>The front desk is taking the details.</p>}
          </section>
        ) : (
          <section className="lw-paper lw-sheet" aria-label="Answers">
            <div className="lw-minutes-h"><h2 className="lw-h2" style={{ margin: 0 }}>Answers</h2><ScreeningPill s={i.screening} /></div>
            <p className="lw-meta" style={{ margin: '0 0 12px' }}>{i.screening.decision ? (i.screening.decision === 'cleared' ? 'The conflicts team cleared this enquiry.' : 'The conflicts team found a conflict. Reply only with "We cannot help", which gives no reason.') : i.screening.message}</p>
            <AnswersList script={i.script} answers={i.answers} />
          </section>
        )}

        {matches && i.status !== 'purged' && i.status !== 'open' && i.status !== 'invited' && (
          <ConflictsPanel matches={matches} decision={i.screening.decision} busy={busy}
            onDecide={(decision, note) => call(() => api.post(`${base}/decision`, { decision, note }), 'The decision could not be recorded.')} />
        )}

        {staffChannel && can.manage && i.status !== 'purged' && <Recording i={i} base={base} busy={busy} call={call} />}

        {can.manage && !['open', 'invited', 'purged'].includes(i.status) && <Replies i={i} base={base} can={can} busy={busy} call={call} />}

        {can.convert && i.status === 'cleared' && <Convert i={i} base={base} onDone={async () => { await qc.invalidateQueries({ queryKey: keys.matters }); await refresh(); }} />}
        {i.status === 'converted' && i.matter_number && (
          <section className="lw-paper lw-sheet" aria-label="Matter opened">
            <p style={{ margin: 0 }}>This enquiry became matter <Link to={`/m/${encodeURIComponent(i.matter_number)}/overview`}><b className="lw-tnum">{i.matter_number}</b></Link>. Members of that matter can open it.</p>
          </section>
        )}

        <section className="lw-paper lw-sheet" aria-label="Record">
          <div className="lw-minutes-h">
            <h2 className="lw-h2" style={{ margin: 0 }}>Record</h2>
            <div style={{ display: 'flex', gap: 6 }}>
              {can.manage && !['converted', 'closed', 'purged'].includes(i.status) && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={busy}
                onClick={() => call(() => api.post(`${base}/close`, {}), 'The enquiry could not be closed.')}><X />Close enquiry</button>}
              {can.purge && i.status !== 'purged' && <Purge base={base} onDone={refresh} />}
            </div>
          </div>
          <ol className="lw-check-lines" aria-label="What was done">
            {i.events.map((e, n) => (
              <li key={n}><span>{ACTION[e.action] ?? e.action}</span><span className="lw-meta">{`${when(e.at)}${e.actor ? ` · ${e.actor}` : ''}`}{e.note ? ` · ${e.note}` : ''}</span></li>
            ))}
          </ol>
        </section>
      </div>
    </>
  );
}

function ConflictsPanel({ matches, decision, busy, onDecide }: {
  matches: IntakeMatch[]; decision: string | null; busy: boolean; onDecide: (d: 'cleared' | 'conflict', note: string) => Promise<unknown>;
}) {
  const [note, setNote] = useState('');
  return (
    <section className="lw-paper lw-sheet" aria-label="Conflicts check">
      <div className="lw-minutes-h">
        <h2 className="lw-h2" style={{ margin: 0 }}>Conflicts check</h2>
        <StatusPill tone="privileged">Conflicts team only</StatusPill>
      </div>
      {matches.length === 0 ? <p style={{ margin: '0 0 12px' }}>The automatic check matched no client or matter party. Confirm against anything it cannot see before clearing.</p> : (
        <div style={{ overflowX: 'auto', marginBottom: 12 }}>
          <table className="lw-table" aria-label="Matches">
            <thead><tr><th>Name given</th><th>Matches</th><th>As</th><th>Client</th><th>Matter</th><th>Likeness</th></tr></thead>
            <tbody>{matches.map((m, n) => (
              <tr key={n}><td>{m.searched}</td><td>{m.matched}</td><td>{KIND[m.kind]}</td><td>{m.client ?? '—'}</td><td className="lw-tnum">{m.matter_number ?? '—'}</td>
                <td className="lw-tnum">{`${Math.round(m.score * 100)}%`}</td></tr>))}</tbody>
          </table>
        </div>
      )}
      <Field label="Note for the record" hint="Seen only by the conflicts team and managing partners; removed if the enquiry is purged.">
        {(id) => <textarea id={id} className="lw-input" rows={2} maxLength={1000} value={note} onChange={(e) => setNote(e.target.value)} />}
      </Field>
      <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
        <button type="button" className="lw-btn lw-btn-primary" disabled={busy} onClick={() => onDecide('cleared', note)}><ShieldCheck />{decision === 'cleared' ? 'Cleared' : 'Clear'}</button>
        <button type="button" className="lw-btn" disabled={busy} onClick={() => onDecide('conflict', note)}><ShieldX />Confirm a conflict</button>
      </div>
    </section>
  );
}

function Recording({ i, base, busy, call }: { i: Intake; base: string; busy: boolean; call: (fn: () => Promise<unknown>, f: string) => Promise<boolean> }) {
  const [method, setMethod] = useState<'verbal' | 'written'>('verbal');
  return (
    <section className="lw-paper lw-sheet" aria-label="Call recording">
      <h2 className="lw-h2" style={{ marginTop: 0 }}>Call recording</h2>
      {!i.consent ? (
        <div style={{ display: 'grid', gap: 10 }}>
          <p style={{ margin: 0 }}>A call is recorded only with the caller's consent. Ask: “May we record this call so that we have an accurate note of your enquiry?”</p>
          <div style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
            <Field label="How the caller agreed">{(id) => (
              <select id={id} className="lw-input" value={method} onChange={(e) => setMethod(e.target.value as 'verbal' | 'written')}>
                <option value="verbal">Said yes on the call</option><option value="written">Agreed in writing</option>
              </select>)}</Field>
            <button type="button" className="lw-btn" disabled={busy} onClick={() => call(() => api.post(`${base}/consent`, { method }), 'The consent could not be recorded.')}><Check />Record consent</button>
          </div>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: 10 }}>
          <p className="lw-meta" style={{ margin: 0 }}>{`Consent ${i.consent.method === 'verbal' ? 'given on the call' : 'given in writing'}, recorded ${when(i.consent.recorded_at)}${i.consent.recorded_by ? ` by ${i.consent.recorded_by}` : ''}.`}</p>
          {i.recording ? <audio controls src={`${base}/recording`} aria-label="Call recording" style={{ width: '100%' }} />
            : <p style={{ margin: 0 }}>No recording attached yet.</p>}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <label className="lw-btn lw-btn-sm"><Mic />{i.recording ? 'Replace recording' : 'Attach recording'}
              <input type="file" accept="audio/*" hidden onChange={(e) => {
                const f = e.target.files?.[0];
                if (!f) return;
                const form = new FormData();
                form.append('file', f);
                void call(() => api.upload(`${base}/recording`, form), 'The recording could not be attached.');
                e.target.value = '';
              }} /></label>
            <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" disabled={busy}
              onClick={() => call(() => api.post(`${base}/consent/withdraw`, {}), 'The withdrawal could not be recorded.')}>The caller withdraws consent</button>
          </div>
          <p className="lw-meta" style={{ margin: 0 }}>If the caller withdraws consent, recording stops and the recording is deleted.</p>
        </div>
      )}
    </section>
  );
}

function Replies({ i, base, can, busy, call }: { i: Intake; base: string; can: IntakeCan; busy: boolean; call: (fn: () => Promise<unknown>, f: string) => Promise<boolean> }) {
  const me = useSession();
  const [template, setTemplate] = useState<IntakeTemplate>('acknowledge');
  const [request, setRequest] = useState('');
  const [approving, setApproving] = useState<IntakeReply | null>(null);
  const [read, setRead] = useState(false);
  const shown = i.replies.filter((r) => r.status !== 'withdrawn');
  const blocked = template === 'can_help' && i.screening.decision !== 'cleared';
  return (
    <section className="lw-paper lw-sheet" aria-label="Replies">
      <h2 className="lw-h2" style={{ marginTop: 0 }}>Reply to the enquirer</h2>
      <p className="lw-meta" style={{ margin: '0 0 10px' }}>From the firm's wording only. A lawyer who did not write it approves it before the enquirer sees it{i.channel === 'link' ? ' on their link page' : ''}. A reply never names another client or party, and "We cannot help" gives no reason.</p>
      {shown.map((r) => (
        <article key={r.id} className="lw-intake-reply" aria-label={`${TEMPLATE[r.template]} — ${r.status}`}>
          <div className="lw-minutes-h">
            <b>{TEMPLATE[r.template]}</b>
            <StatusPill tone={r.status === 'released' ? 'verified' : 'attention'}>{r.status === 'released' ? `Approved by ${r.approved_by}` : `Draft by ${r.drafted_by} — awaiting a lawyer`}</StatusPill>
          </div>
          <p style={{ whiteSpace: 'pre-wrap', margin: '6px 0' }}>{r.body}</p>
          {r.status === 'released' && i.channel !== 'link' && <CopyField label="Text to send" value={r.body} />}
          {r.status === 'draft' && can.approve && r.drafted_by !== me.user.username && <button type="button" className="lw-btn lw-btn-sm" onClick={() => { setRead(false); setApproving(r); }}><Check />Approve and release</button>}
          {r.status === 'draft' && can.approve && r.drafted_by === me.user.username && <p className="lw-meta" style={{ margin: 0 }}>You wrote this reply, so another lawyer approves it.</p>}
        </article>
      ))}
      <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
        <Field label="New reply">{(id) => (
          <select id={id} className="lw-input" value={template} onChange={(e) => setTemplate(e.target.value as IntakeTemplate)}>
            {(Object.keys(TEMPLATE) as IntakeTemplate[]).map((t) => <option key={t} value={t}>{TEMPLATE[t]}</option>)}
          </select>)}</Field>
        {template === 'more_info' && <Field label="What we need" hint="Plain words; no names of other clients or parties.">
          {(id) => <textarea id={id} className="lw-input" rows={2} maxLength={600} value={request} onChange={(e) => setRequest(e.target.value)} />}</Field>}
        {blocked && <p className="lw-meta" style={{ margin: 0 }}>The conflicts team must clear the enquiry before we offer to help.</p>}
        <div><button type="button" className="lw-btn" disabled={busy || blocked || (template === 'more_info' && !request.trim())}
          onClick={async () => { if (await call(() => api.post(`${base}/replies`, { template, request: template === 'more_info' ? request : undefined }), 'The reply could not be drafted.')) setRequest(''); }}>Draft reply</button></div>
      </div>
      {approving && (
        <Dialog title="Approve this reply?" onClose={() => setApproving(null)}
          footer={<><button type="button" className="lw-btn" onClick={() => setApproving(null)}>Cancel</button>
            <button type="button" className="lw-btn lw-btn-primary" disabled={!read || busy} onClick={async () => {
              if (await call(() => api.post(`${base}/replies/${encodeURIComponent(approving.id)}/approve`, { body_sha256: approving.body_sha256 }), 'The reply could not be approved.')) setApproving(null);
            }}><Check />Approve and release</button></>}>
          <p style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{approving.body}</p>
          <label className="lw-meta" style={{ display: 'flex', gap: 8, marginTop: 10 }}><input type="checkbox" checked={read} onChange={(e) => setRead(e.target.checked)} />I have read this reply and it gives no advice or other client's details</label>
        </Dialog>
      )}
    </section>
  );
}

function Convert({ i, base, onDone }: { i: Intake; base: string; onDone: () => Promise<void> }) {
  const [open, setOpen] = useState(false);
  const org = typeof i.answers.organisation === 'string' && i.answers.organisation ? i.answers.organisation : String(i.answers.your_name ?? '');
  const [form, setForm] = useState({ client_name: org, client_code: '', matter_number: '', matter_title: '' });
  const [error, setError] = useState('');
  const submit = async () => {
    setError('');
    try {
      await api.post<IntakeView & { matter: { number: string } }>(`${base}/convert`, { ...form, client_code: form.client_code || undefined });
      setOpen(false);
      await onDone();
    } catch (e) { setError(message(e, 'The matter could not be opened.')); }
  };
  return (
    <section className="lw-paper lw-sheet" aria-label="Open a matter">
      <div className="lw-minutes-h">
        <div><h2 className="lw-h2" style={{ margin: 0 }}>Open a matter</h2><p className="lw-meta" style={{ margin: '2px 0 0' }}>The client and the matter are created, you become its lead, and the other parties named go on its party register.</p></div>
        <button type="button" className="lw-btn lw-btn-primary" onClick={() => setOpen(true)}><FolderPlus />Open a matter</button>
      </div>
      {open && (
        <Dialog title="Open a matter from this enquiry" onClose={() => setOpen(false)}
          footer={<><button type="button" className="lw-btn" onClick={() => setOpen(false)}>Cancel</button>
            <button type="button" className="lw-btn lw-btn-primary" disabled={!form.client_name.trim() || !form.matter_number.trim() || !form.matter_title.trim()} onClick={submit}><FolderPlus />Open the matter</button></>}>
          <div style={{ display: 'grid', gap: 10 }}>
            <Field label="Client">{(id) => <input id={id} className="lw-input" value={form.client_name} onChange={(e) => setForm({ ...form, client_name: e.target.value })} />}</Field>
            <Field label="Client code (optional)">{(id) => <input id={id} className="lw-input" value={form.client_code} onChange={(e) => setForm({ ...form, client_code: e.target.value })} />}</Field>
            <Field label="Matter number">{(id) => <input id={id} className="lw-input lw-tnum" value={form.matter_number} onChange={(e) => setForm({ ...form, matter_number: e.target.value })} />}</Field>
            <Field label="Matter title">{(id) => <input id={id} className="lw-input" value={form.matter_title} onChange={(e) => setForm({ ...form, matter_title: e.target.value })} />}</Field>
            {error && <ErrorNote>{error}</ErrorNote>}
          </div>
        </Dialog>
      )}
    </section>
  );
}

function Purge({ base, onDone }: { base: string; onDone: () => Promise<void> }) {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState('');
  return (
    <>
      <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setOpen(true)}><Trash2 />Purge personal data</button>
      {open && (
        <Dialog title="Purge this enquiry's personal data?" onClose={() => setOpen(false)}
          footer={<><button type="button" className="lw-btn" onClick={() => setOpen(false)}>Cancel</button>
            <button type="button" className="lw-btn lw-btn-primary" onClick={async () => {
              try { await api.del(base); setOpen(false); await onDone(); } catch (e) { setError(message(e, 'The enquiry could not be purged.')); }
            }}><Trash2 />Purge</button></>}>
          <p style={{ margin: 0 }}>The answers, the screening matches, the replies, the recording and the conflicts notes are deleted. The record of what was done, by whom and when, stays. This cannot be undone.</p>
          {error && <ErrorNote>{error}</ErrorNote>}
        </Dialog>
      )}
    </>
  );
}
