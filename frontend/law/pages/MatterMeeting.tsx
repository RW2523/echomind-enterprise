import React, { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router';
import { Circle, FileOutput, Mic, Pause, Play, Square, Trash2, UserPlus } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys, meetingsPath, useMeeting } from '../api/queries';
import type { CaptureRecord, ConsentMethod, MeetingDetail, Participant, ParticipantRole, Segment, TranscriptData } from '../api/types';
import { useMatter } from '../app/matter';
import { startMic, type Mic as MicHandle } from '../app/mic';
import Toolbar from '../ui/Toolbar';
import { Dialog, ErrorNote, Field, Skeleton, StatusPill, type Tone } from '../ui/primitives';
import NotAvailable from './NotAvailable';
import { DeleteDialog, HoldNotice, LabelControl } from '../ui/records';
import { kindLabel } from './MatterMeetings';
import { ExportDialog } from './MatterExports';
import { TranscriptRecord } from '../ui/transcript';
import { AlertsPanel } from '../ui/alerts';
import { useAlerts } from '../api/queries';
import type { AssistAlert } from '../api/types';

const ROLES: [ParticipantRole, string][] = [['host', 'Host'], ['attorney', 'Attorney'], ['client', 'Client'], ['counterparty', 'Counterparty'],
  ['witness', 'Witness'], ['expert', 'Expert'], ['other', 'Other']];
const roleText = (r: string) => ROLES.find(([v]) => v === r)?.[1] ?? r;
const METHODS: Record<ConsentMethod, string> = { verbal_on_record: 'Spoken, on the record', written: 'In writing', host_attestation: 'Confirmed by the host' };
const CONSENT: Record<Participant['consent']['state'], [string, Tone]> = {
  granted: ['Consented', 'verified'], missing: ['Needs consent', 'attention'], withdrawn: ['Withdrawn', 'conflict'], stale: ['Expired — ask again', 'attention'],
};
const STOPPED: Record<string, string> = {
  ended: 'Recording stopped.',
  consent_withdrawn: 'Recording stopped: a participant withdrew consent. Nothing after that moment was captured.',
  access_revoked: 'Recording stopped: your access to this matter changed.',
  signed_out: 'Recording stopped: you were signed out.',
  meeting_ended: 'Recording stopped: the meeting was ended.',
  frame_too_large: 'Recording stopped: the browser sent audio in a form this server does not accept.',
};
const REFUSED: Record<number, string> = {
  4412: 'Everyone must consent before recording can start.', 4403: 'You cannot record in this matter.',
  4409: 'This matter or meeting is closed.', 4404: 'This meeting is not available.', 1008: 'Your session could not be verified. Sign in again.',
};
const time = (iso?: string | null) => (iso ? new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '');
const day = (iso?: string | null) => (iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '');

type Tab = 'prepare' | 'consent' | 'record';

/** One meeting: who takes part, their consent, and consent-gated recording. */
export default function MatterMeeting() {
  const matter = useMatter();
  const { meetingId = '' } = useParams();
  const meeting = useMeeting(matter.id, meetingId);
  const [tab, setTab] = useState<Tab>('consent');
  if (meeting.isPending) return <div className="lw-content" aria-busy="true"><Skeleton height={48} /><Skeleton height={200} /></div>;
  if (meeting.isError) return <NotAvailable />;
  const m = meeting.data;
  const contributor = ['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open' && m.status === 'planned';
  const waiting = m.participants.filter((p) => p.consent.state !== 'granted');
  const tabs: [Tab, string][] = [['prepare', 'Prepare'], ['consent', `Consent · ${m.consent.granted} of ${m.consent.needed}`], ['record', 'Record']];
  return (
    <>
      <Toolbar crumb={`${matter.number} · Meetings`} title={m.title} />
      <div className="lw-content" style={{ maxWidth: 1000 }}>
        <div className="lw-seg" role="tablist" aria-label="Meeting">
          {tabs.map(([id, label]) => (
            <button key={id} type="button" role="tab" id={`tab-${id}`} aria-selected={tab === id} aria-controls={`panel-${id}`}
              className="lw-seg-item" onClick={() => setTab(id)}>{label}</button>
          ))}
        </div>
        <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} style={{ display: 'grid', gap: 16 }}>
          {tab === 'prepare' && <Prepare m={m} matterId={matter.id} canEdit={contributor}
            canSetLabel={['lead', 'attorney', 'reviewer'].includes(matter.my_role ?? '') && matter.status === 'open'}
            canSuggestLabel={matter.my_role === 'paralegal' && matter.status === 'open'} />}
          {tab === 'consent' && <Consent m={m} matterId={matter.id} canRecord={contributor} />}
          {tab === 'record' && <Record m={m} matterId={matter.id} canRecord={contributor} waiting={waiting}
            canDelete={['lead', 'attorney'].includes(matter.my_role ?? '')} onHold={!!matter.on_hold}
            canCorrect={['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && matter.status === 'open'} />}
        </div>
      </div>
    </>
  );
}

function useRefresh(matterId: string, meetingId: string) {
  const qc = useQueryClient();
  return () => Promise.all([qc.invalidateQueries({ queryKey: keys.meeting(matterId, meetingId) }), qc.invalidateQueries({ queryKey: keys.meetings(matterId) })]);
}

// ── Prepare ────────────────────────────────────────────────────────────────────
function Prepare({ m, matterId, canEdit, canSetLabel, canSuggestLabel }: { m: MeetingDetail; matterId: string; canEdit: boolean; canSetLabel: boolean; canSuggestLabel: boolean }) {
  const refresh = useRefresh(matterId, m.id);
  const [form, setForm] = useState({ name: '', role: 'client' as ParticipantRole });
  const [error, setError] = useState('');
  const base = `${meetingsPath(matterId)}/${encodeURIComponent(m.id)}`;
  const add = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try { await api.post(`${base}/participants`, { name: form.name.trim(), role: form.role }); setForm({ name: '', role: 'client' }); await refresh(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The participant could not be added.'); }
  };
  const remove = async (p: Participant) => {
    setError('');
    try { await api.del(`${base}/participants/${encodeURIComponent(p.id)}`); await refresh(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The participant could not be removed.'); }
  };
  return (
    <section className="lw-paper lw-sheet" aria-label="Participants" style={{ display: 'grid', gap: 14 }}>
      <div>
        <div className="lw-eyebrow">{kindLabel(m.kind)}{m.scheduled_at ? ` · ${day(m.scheduled_at)}` : ''}</div>
        <h2 className="lw-h2" style={{ marginTop: 4 }}>Who takes part</h2>
        <p className="lw-meta" style={{ margin: '4px 0 0' }}>List everyone whose voice may be recorded. Someone who joins later must be added and consent before recording continues.</p>
      </div>
      <ul className="lw-people">
        {m.participants.map((p) => (
          <li key={p.id}>
            <span style={{ fontWeight: 600 }}>{p.name}</span><span className="lw-meta">{roleText(p.role)}</span>
            {canEdit && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" style={{ marginLeft: 'auto' }} onClick={() => remove(p)} aria-label={`Remove ${p.name}`}>Remove</button>}
          </li>
        ))}
        {m.participants.length === 0 && <li className="lw-meta">No participants yet.</li>}
      </ul>
      {canEdit && (
        <form onSubmit={add} style={{ display: 'grid', gridTemplateColumns: '1fr 180px auto', gap: 10, alignItems: 'end' }}>
          <Field label="Name">{(id) => <input id={id} className="lw-input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />}</Field>
          <Field label="Role">{(id) => <select id={id} className="lw-input" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as ParticipantRole })}>
            {ROLES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>}</Field>
          <button type="submit" className="lw-btn" disabled={!form.name.trim()}><UserPlus />Add</button>
        </form>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
      <AssistChoice m={m} base={base} canEdit={canEdit} onDone={refresh} />
      <div style={{ borderTop: '1px solid var(--rule)', paddingTop: 14 }}>
        <h3 className="lw-eyebrow" style={{ margin: '0 0 6px' }}>Privilege label</h3>
        <LabelControl matterId={matterId} type="meeting" objectId={m.id} canSet={canSetLabel} canSuggest={canSuggestLabel} />
      </div>
    </section>
  );
}

const PACKS: [string, string][] = [['', 'The default for this kind of meeting'], ['legal_client_call', 'Client call'],
  ['legal_negotiation', 'Negotiation'], ['legal_governance', 'Board or governance'], ['off', 'Off — no alerts']];

/** Which Silent Assist pack listens to this meeting (S2.5). */
function AssistChoice({ m, base, canEdit, onDone }: { m: MeetingDetail; base: string; canEdit: boolean; onDone: () => void }) {
  const [error, setError] = useState('');
  const set = async (pack: string) => {
    setError('');
    try { await api.patch(`${base}/assist`, { pack: pack || null }); onDone(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The choice could not be saved.'); }
  };
  return (
    <div style={{ borderTop: '1px solid var(--rule)', paddingTop: 14 }}>
      <h3 className="lw-eyebrow" style={{ margin: '0 0 6px' }}>Silent Assist</h3>
      <p className="lw-meta" style={{ margin: '0 0 8px' }}>While recording, what is said is compared with this matter’s documents and flagged for review: figures that differ from the signed terms or the board’s authority, commitments and privileged remarks with a counterparty on the call, actions, quorum and conflicts of interest.</p>
      {canEdit ? (
        <Field label="Pack">{(id) => (
          <select id={id} className="lw-input" style={{ maxWidth: 360 }} value={m.assist?.chosen ?? ''} onChange={(e) => set(e.target.value)}>
            {PACKS.map(([v, l]) => <option key={v} value={v}>{v === '' && m.assist?.pack && !m.assist?.chosen ? `${l} (${m.assist.label})` : l}</option>)}
          </select>
        )}</Field>
      ) : <p className="lw-meta" style={{ margin: 0 }}>{m.assist?.pack ? `${m.assist.label} pack` : 'Off'}</p>}
      {error && <ErrorNote>{error}</ErrorNote>}
    </div>
  );
}

// ── Consent ────────────────────────────────────────────────────────────────────
function Consent({ m, matterId, canRecord }: { m: MeetingDetail; matterId: string; canRecord: boolean }) {
  const refresh = useRefresh(matterId, m.id);
  const [asking, setAsking] = useState<Participant | null>(null);
  const [error, setError] = useState('');
  const base = `${meetingsPath(matterId)}/${encodeURIComponent(m.id)}/participants`;
  const withdraw = async (p: Participant) => {
    setError('');
    try { await api.post(`${base}/${encodeURIComponent(p.id)}/consent/withdraw`); await refresh(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The withdrawal could not be recorded.'); }
  };
  return (
    <>
      <section className="lw-paper lw-sheet" aria-label="Consent statement" style={{ display: 'grid', gap: 8 }}>
        <h2 className="lw-h2">Read this to each participant</h2>
        <blockquote className="lw-serif lw-statement">{m.policy.statement}</blockquote>
        <p className="lw-meta" style={{ margin: 0 }}>Everyone must agree before recording starts, and consent older than {m.policy.max_age_hours} hours must be asked again. Wording and rules: firm policy pending counsel review.</p>
      </section>
      <section className="lw-paper" aria-label="Consent">
        <table className="lw-table">
          <thead><tr><th>Participant</th><th>Consent</th><th><span className="sr-only">Actions</span></th></tr></thead>
          <tbody>
            {m.participants.map((p) => {
              const [label, tone] = CONSENT[p.consent.state];
              return (
                <tr key={p.id}>
                  <td><div style={{ fontWeight: 600 }}>{p.name}</div><div className="lw-meta">{roleText(p.role)}</div></td>
                  <td><StatusPill tone={tone}>{label}</StatusPill>
                    {p.consent.at && <div className="lw-meta">{p.consent.method ? `${METHODS[p.consent.method]} · ` : ''}{day(p.consent.at)}{p.consent.by ? ` · recorded by ${p.consent.by}` : ''}</div>}</td>
                  <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                    {canRecord && p.consent.state !== 'granted' && <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" onClick={() => setAsking(p)}>Record consent</button>}
                    {p.consent.state === 'granted' && <button type="button" className="lw-btn lw-btn-sm lw-btn-danger" onClick={() => withdraw(p)}>Withdraw</button>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {m.participants.length === 0 && <p className="lw-meta" style={{ padding: 20, margin: 0 }}>Add participants first (Prepare).</p>}
      </section>
      {error && <ErrorNote>{error}</ErrorNote>}
      {asking && <ConsentDialog m={m} matterId={matterId} p={asking} onClose={() => setAsking(null)} onSaved={refresh} />}
    </>
  );
}

function ConsentDialog({ m, matterId, p, onClose, onSaved }: { m: MeetingDetail; matterId: string; p: Participant; onClose: () => void; onSaved: () => void }) {
  const [method, setMethod] = useState<ConsentMethod>('verbal_on_record');
  const [read, setRead] = useState(false);
  const [error, setError] = useState('');
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try {
      await api.post(`${meetingsPath(matterId)}/${encodeURIComponent(m.id)}/participants/${encodeURIComponent(p.id)}/consent`, { method, statement_read: read });
      onSaved(); onClose();
    } catch (err) { setError(err instanceof ApiError ? err.message : 'The consent could not be recorded.'); }
  };
  return (
    <Dialog title={`Consent of ${p.name}`} onClose={onClose}>
      <form onSubmit={submit} style={{ display: 'grid', gap: 12 }}>
        <blockquote className="lw-serif lw-statement" style={{ margin: 0 }}>{m.policy.statement}</blockquote>
        <Field label="How was consent given?">{(id) => <select id={id} className="lw-input" value={method} onChange={(e) => setMethod(e.target.value as ConsentMethod)}>
          {m.policy.methods.map((v) => <option key={v} value={v}>{METHODS[v]}</option>)}</select>}</Field>
        <label style={{ display: 'flex', gap: 8, alignItems: 'flex-start', fontSize: 14 }}>
          <input type="checkbox" checked={read} onChange={(e) => setRead(e.target.checked)} style={{ marginTop: 3 }} />
          <span>I read the statement to {p.name}, and they agreed to be recorded.</span>
        </label>
        {error && <ErrorNote>{error}</ErrorNote>}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="lw-btn lw-btn-primary" disabled={!read}>Record consent</button>
        </div>
      </form>
    </Dialog>
  );
}

// ── Record ─────────────────────────────────────────────────────────────────────
type Phase = 'idle' | 'connecting' | 'live' | 'paused' | 'stopped';

function Record({ m, matterId, canRecord, waiting, canDelete, onHold, canCorrect }: {
  m: MeetingDetail & { transcript?: Segment[]; transcriptData?: TranscriptData }; matterId: string; canRecord: boolean; waiting: Participant[];
  canDelete: boolean; onHold: boolean; canCorrect: boolean;
}) {
  const refresh = useRefresh(matterId, m.id);
  const [deleting, setDeleting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const matter = useMatter();
  const transcriptBase = `${meetingsPath(matterId)}/${encodeURIComponent(m.id)}/transcript`;
  const [phase, setPhase] = useState<Phase>('idle');
  const [note, setNote] = useState('');
  const [transcription, setTranscription] = useState<'available' | 'unavailable' | null>(null);
  const [live, setLive] = useState<Segment[]>([]);
  const [started, setStarted] = useState<number | null>(null);
  const [speaker, setSpeaker] = useState<string | null>(null);
  const [liveAlerts, setLiveAlerts] = useState<AssistAlert[]>([]);
  const alerts = useAlerts(matterId, m.id);
  const [warning, setWarning] = useState('');
  const [, tick] = useState(0);
  const ws = useRef<WebSocket | null>(null);
  const mic = useRef<MicHandle | null>(null);
  const ready = waiting.length === 0 && m.participants.length > 0;

  useEffect(() => {
    if (phase !== 'live' && phase !== 'paused') return;
    const t = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [phase]);
  useEffect(() => () => { mic.current?.stop(); ws.current?.close(); }, []);

  const end = (message: string) => {
    mic.current?.stop(); mic.current = null;
    setPhase('stopped'); setNote(message); setStarted(null);
    void refresh(); void alerts.refetch();
  };
  const start = () => {
    setNote(''); setLive([]); setPhase('connecting'); setSpeaker(null); setWarning(''); setLiveAlerts([]);
    const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const sock = new WebSocket(`${scheme}://${window.location.host}${meetingsPath(matterId)}/${encodeURIComponent(m.id)}/capture?sample_rate=16000`);
    sock.binaryType = 'arraybuffer';
    ws.current = sock;
    let stopped = false;
    sock.onmessage = async (e) => {
      const msg = JSON.parse(String(e.data));
      if (msg.type === 'ready') {
        setTranscription(msg.transcription);
        try {
          mic.current = await startMic((pcm) => { if (sock.readyState === WebSocket.OPEN) sock.send(pcm); });
          setPhase('live'); setStarted(Date.now());
        } catch {
          sock.send(JSON.stringify({ type: 'stop' }));
          end('The browser refused microphone access, so nothing was recorded.');
          stopped = true;
        }
      } else if (msg.type === 'segment') setLive((s) => [...s, msg]);
      else if (msg.type === 'speaker') setSpeaker(msg.participant_id ?? null);
      else if (msg.type === 'degraded') setWarning(msg.reason === 'behind'
        ? `Transcription is ${msg.lag_s} s behind. Speech is still being received; the transcript will catch up, and this is noted in the record.`
        : 'A transcription error left a gap. It is marked in the transcript at this point, so nothing is lost silently.');
      else if (msg.type === 'recovered') setWarning('');
      else if (msg.type === 'alert') setLiveAlerts((xs) => [...xs, msg as AssistAlert]);
      else if (msg.type === 'state') setPhase(msg.state === 'paused' ? 'paused' : 'live');
      else if (msg.type === 'stopped') { stopped = true; end(STOPPED[msg.reason] ?? 'Recording stopped.'); }
    };
    sock.onclose = (e) => { if (!stopped) end(REFUSED[e.code] ?? 'The recording connection closed.'); stopped = true; };
  };
  const send = (type: string) => ws.current?.send(JSON.stringify({ type }));
  const speaking = (id: string | null) => ws.current?.send(JSON.stringify({ type: 'speaker', participant_id: id }));
  const speakers = m.participants.filter((x) => x.consent.state === 'granted');
  useEffect(() => {                                 // Alt+1…9 marks who is speaking, Alt+0 nobody in particular
    if (phase !== 'live') return;
    const onKey = (e: KeyboardEvent) => {
      if (!e.altKey || !/^Digit\d$/.test(e.code)) return;
      const k = Number(e.code.slice(5));
      if (k === 0) { e.preventDefault(); speaking(null); } else if (speakers[k - 1]) { e.preventDefault(); speaking(speakers[k - 1].id); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });
  const elapsed = started ? Math.floor((Date.now() - started) / 1000) : 0;
  const clock = `${String(Math.floor(elapsed / 60)).padStart(2, '0')}:${String(elapsed % 60).padStart(2, '0')}`;
  const recording = phase === 'live' || phase === 'paused';

  return (
    <>
      <section className="lw-paper lw-sheet lw-capsule" aria-label="Recording" data-phase={phase}>
        {!recording && phase !== 'connecting' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <Mic size={22} style={{ color: ready ? 'var(--accent)' : 'var(--ink-3)', flex: 'none' }} />
            <div style={{ flex: 1 }}>
              <h2 className="lw-h2">{m.status === 'ended' ? 'This meeting has ended' : ready ? 'Everyone has consented' : 'Waiting for consent'}</h2>
              <p className="lw-meta" style={{ margin: '2px 0 0' }}>{m.status === 'ended'
                ? 'Nothing more can be recorded. The transcript below can still be corrected, with reasons, and added to research.'
                : ready ? 'Recording can start. Anyone on the matter can stop it by recording a withdrawal.'
                : `Still needed from: ${waiting.map((p) => p.name).join(', ') || 'add participants first'}.`}</p>
            </div>
            {canRecord && <button type="button" className="lw-btn lw-btn-primary" disabled={!ready} onClick={start}><Circle />Start recording</button>}
          </div>
        )}
        {phase === 'connecting' && <p className="lw-meta" style={{ margin: 0 }}>Checking consent and opening the recording…</p>}
        {recording && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <span className={`lw-rec-dot${phase === 'paused' ? ' lw-rec-paused' : ''}`} aria-hidden="true" />
            <div style={{ flex: 1 }} role="status" aria-live="polite">
              <div style={{ fontWeight: 650 }}>{phase === 'paused' ? 'Paused — nothing is being captured' : 'Recording'} <span className="lw-tnum lw-meta">{clock}</span></div>
              {transcription === 'unavailable' && <div className="lw-meta">Live transcription is not available on this server. Consent, pauses and audio receipts are still recorded.</div>}
            </div>
            {phase === 'live' ? <button type="button" className="lw-btn" onClick={() => send('pause')}><Pause />Pause</button>
              : <button type="button" className="lw-btn" onClick={() => send('resume')}><Play />Resume</button>}
            <button type="button" className="lw-btn lw-btn-danger" onClick={() => send('stop')}><Square />Stop</button>
          </div>
        )}
        {recording && transcription === 'available' && speakers.length > 0 && (
          <fieldset className="lw-segment lw-speaking" aria-label="Speaking now">
            <legend className="lw-sr">Speaking now (Alt+1 to Alt+9)</legend>
            {[...speakers.map((x, i) => [x.id, x.name, `Alt+${i + 1}`] as const), [null, 'Not marked', 'Alt+0'] as const].map(([id, name, key]) => (
              <label key={id ?? 'none'} data-on={speaker === id || undefined} title={key}>
                <input type="radio" name="speaking" checked={speaker === id} onChange={() => speaking(id)} />{name}
              </label>
            ))}
          </fieldset>
        )}
        {warning && recording && <p role="alert" className="lw-capture-warning">{warning}</p>}
        {note && <p role="alert" className="lw-meta" style={{ margin: recording ? '10px 0 0' : '12px 0 0', fontSize: 13.5, color: 'var(--ink-2)' }}>{note}</p>}
      </section>
      {(recording ? m.assist?.pack : (alerts.data?.alerts.length ?? 0) > 0) && (
        <AlertsPanel alerts={recording ? liveAlerts : alerts.data?.alerts ?? []} assist={recording ? m.assist : alerts.data?.assist}
          telemetry={alerts.data?.telemetry} matterId={matterId} meetingId={m.id} canDecide={canCorrect} live={recording} />
      )}
      {(live.length > 0 || (m.transcript?.length ?? 0) > 0) && (
        <section className="lw-paper lw-sheet" aria-label="Transcript">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
            <h2 className="lw-h2" style={{ marginRight: 'auto' }}>Transcript</h2>
            {['lead', 'attorney', 'paralegal'].includes(matter.my_role ?? '') && !recording && (m.transcript?.length ?? 0) > 0 &&
              <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => setExporting(true)}><FileOutput />Export…</button>}
            {canDelete && !onHold && !recording && (m.transcript?.length ?? 0) > 0 &&
              <button type="button" className="lw-btn lw-btn-sm lw-btn-danger" onClick={() => setDeleting(true)}><Trash2 />Delete transcript…</button>}
          </div>
          {canDelete && onHold && <HoldNotice />}
          {m.transcriptData && (
            <TranscriptRecord matterId={matterId} meetingId={m.id} data={m.transcriptData} canCorrect={canCorrect && !recording}
              canFinalise={canCorrect && m.status === 'ended' && !recording && (m.transcript?.length ?? 0) > 0}
              live={recording ? live : undefined} />
          )}
        </section>
      )}
      {m.captures.length > 0 && <Timeline captures={m.captures} />}
      {exporting && <ExportDialog matterId={matterId} number={matter.number} objectType="meeting_transcript" objectId={m.id} title={`Transcript of ${m.title}`} onClose={() => setExporting(false)} />}
      {deleting && (
        <DeleteDialog title="Delete this meeting’s transcript" previewUrl={`${transcriptBase}/deletion-preview`} deleteUrl={`${transcriptBase}/delete`}
          onClose={() => setDeleting(false)} onDone={() => { setDeleting(false); void refresh(); }} />
      )}
    </>
  );
}

const EVENT_TEXT: Record<string, string> = { started: 'Recording started', paused: 'Paused', resumed: 'Resumed', stopped: 'Stopped', frame_rejected: 'Audio refused after stop' };

function Timeline({ captures }: { captures: CaptureRecord[] }) {
  return (
    <section className="lw-paper lw-sheet" aria-label="Recording history">
      <h2 className="lw-h2" style={{ marginBottom: 10 }}>Recording history</h2>
      {captures.map((c) => (
        <div key={c.id} style={{ marginBottom: 12 }}>
          <ol className="lw-timeline">
            {c.events.map((e, i) => (
              <li key={i}><span className="lw-tnum lw-meta">{time(e.at)}</span><span>{EVENT_TEXT[e.kind] ?? e.kind}{e.reason && e.kind === 'stopped' ? ` — ${(STOPPED[e.reason] ?? e.reason).replace(/^Recording stopped:? ?/, '').replace(/\.$/, '') || 'ended'}` : ''}</span>
                {e.by && <span className="lw-meta">{e.by}</span>}</li>
            ))}
          </ol>
          <p className="lw-meta" style={{ margin: '4px 0 0' }}>Consent relied on: <span className="lw-mono">{c.consent_snapshot.slice(0, 12)}</span> · {c.custody_records} audio receipt{c.custody_records === 1 ? '' : 's'} ({Math.round(c.audio_bytes / 1024)} KB received; audio itself not kept) · {c.segments} transcript segment{c.segments === 1 ? '' : 's'}</p>
        </div>
      ))}
    </section>
  );
}
