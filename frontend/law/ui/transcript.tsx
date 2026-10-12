import React, { useState } from 'react';
import { Link } from 'react-router';
import { BookOpenText, Check, PenLine } from 'lucide-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { keys, meetingsPath } from '../api/queries';
import type { CaptureMark, Segment, TranscriptData, VocabularySuggestion } from '../api/types';
import { useMatter } from '../app/matter';
import { Dialog, ErrorNote, Field, StatusPill } from './primitives';

const clock = (iso?: string | null) => (iso ? new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '');
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
const KIND: Record<VocabularySuggestion['kind'], string> = { participant: 'A participant', company: 'A company in the documents', defined_term: 'A defined term' };

type Row = { kind: 'segment'; s: Segment } | { kind: 'gap'; m: CaptureMark };

/** The transcript as a record: each segment with its time and speaker, corrections marked with
 *  their history, gaps where nothing could be transcribed, and matter-vocabulary suggestions. */
export function TranscriptRecord({ matterId, meetingId, data, canCorrect, canFinalise, live }: {
  matterId: string; meetingId: string; data: TranscriptData; canCorrect: boolean; canFinalise: boolean; live?: Segment[];
}) {
  const matter = useMatter();
  const qc = useQueryClient();
  const base = `${meetingsPath(matterId)}/${encodeURIComponent(meetingId)}/transcript`;
  const [editing, setEditing] = useState<Segment | null>(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const suggestions = useQuery({
    queryKey: [...keys.meeting(matterId, meetingId), 'suggestions'], enabled: canCorrect && !live && data.segments.length > 0,
    refetchOnWindowFocus: false,
    queryFn: async () => (await api.get<{ suggestions: VocabularySuggestion[] }>(`${base}/suggestions`)).suggestions,
  });
  const refresh = () => qc.invalidateQueries({ queryKey: keys.meeting(matterId, meetingId) });
  const names = Object.fromEntries((data.participants ?? []).map((p) => [p.id, p.name]));
  const segments = live ?? data.segments;
  const rows: Row[] = [...segments.map((s) => ({ kind: 'segment' as const, s })),
    ...(live ? [] : (data.marks ?? []).filter((m) => m.kind === 'gap').map((m) => ({ kind: 'gap' as const, m })))]
    .sort((a, b) => ((a.kind === 'segment' ? a.s.wall_start : a.m.at) ?? '').localeCompare((b.kind === 'segment' ? b.s.wall_start : b.m.at) ?? ''));

  const accept = async (x: VocabularySuggestion) => {
    setBusy(x.id); setError('');
    const from = x.source.kind === 'document' ? x.source.document_title : 'the meeting’s participants';
    try {
      await api.post(`${base}/segments/${encodeURIComponent(x.segment_id)}/revisions`, { suggestion_id: x.id, reason: `Matter vocabulary: ${x.suggested} (${from})` });
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The correction could not be saved.');
    } finally { setBusy(''); }
  };
  const finalise = async () => {
    setBusy('finalise'); setError('');
    try {
      await api.post(`${base}/finalise`);
      await Promise.all([refresh(), qc.invalidateQueries({ queryKey: keys.documents(matterId) })]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The transcript could not be finalised.');
    } finally { setBusy(''); }
  };

  return (
    <>
      {data.finalised && !live && (
        <p className="lw-meta" style={{ margin: '0 0 10px' }}>
          <StatusPill tone={data.finalised.stale ? 'attention' : 'verified'}>{data.finalised.stale ? 'Corrected since it was added to research' : 'In research'}</StatusPill>{' '}
          Added {when(data.finalised.at)} · <Link className="lw-linklike" to={`/m/${matter.number}/documents/${encodeURIComponent(data.finalised.document_id)}/read`}>Open the finalised transcript</Link>
        </p>
      )}
      {canFinalise && !live && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
          <button type="button" className="lw-btn lw-btn-sm" disabled={busy === 'finalise' || (!!data.finalised && !data.finalised.stale)} onClick={finalise}>
            <BookOpenText />{busy === 'finalise' ? 'Adding…' : data.finalised ? 'Update in research' : 'Add to research'}
          </button>
        </div>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
      <ol className="lw-transcript-rows" aria-label="Transcript segments">
        {rows.map((r) => r.kind === 'gap' ? (
          <li key={`gap-${r.m.at}`} className="lw-transcript-gap">
            <span className="lw-tnum lw-meta">{clock(r.m.at)}</span>
            <span className="lw-meta">{r.m.detail === 'transcription_behind'
              ? 'Not transcribed: speech recognition fell too far behind here.'
              : 'Not transcribed: a transcription error at this point.'}</span>
          </li>
        ) : (
          <li key={r.s.id} className="lw-transcript-row" data-revised={r.s.revised || undefined}>
            <span className="lw-tnum lw-meta">{clock(r.s.wall_start)}</span>
            <div style={{ minWidth: 0 }}>
              <div className="lw-transcript-who">{r.s.speaker ?? (r.s.speaker_participant_id ? names[r.s.speaker_participant_id] : null) ?? 'Unattributed'}</div>
              <p className="lw-serif" style={{ margin: 0 }}>{r.s.text}</p>
              {r.s.revised && <History s={r.s} names={names} />}
            </div>
            {canCorrect && !live && (
              <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" aria-label={`Correct the segment at ${clock(r.s.wall_start) || r.s.seq}`}
                onClick={() => setEditing(r.s)}><PenLine /></button>
            )}
          </li>
        ))}
      </ol>
      {canCorrect && !live && (suggestions.data?.length ?? 0) > 0 && (
        <section aria-label="Matter vocabulary" className="lw-vocab">
          <h3 className="lw-eyebrow" style={{ margin: '14px 0 6px' }}>{`Matter vocabulary · ${suggestions.data!.length}`}</h3>
          <p className="lw-meta" style={{ margin: '0 0 8px' }}>Names this matter already knows, close to what was heard. Nothing changes until you accept.</p>
          <ul className="lw-sources">
            {suggestions.data!.map((x) => (
              <li key={x.id} className="lw-source" style={{ alignItems: 'center' }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div><span className="lw-vocab-heard">{x.heard}</span> → <b>{x.suggested}</b></div>
                  <div className="lw-meta">{KIND[x.kind]}{x.source.kind === 'document' ? ` · ${x.source.document_title}, v${x.source.version_number}` : ''}</div>
                </div>
                <button type="button" className="lw-btn lw-btn-sm" disabled={busy === x.id} onClick={() => accept(x)}
                  aria-label={`Accept ${x.suggested} for ${x.heard}`}><Check />Accept</button>
              </li>
            ))}
          </ul>
        </section>
      )}
      {editing && <CorrectDialog base={base} s={editing} participants={(data.participants ?? []).filter((p) => !p.removed)} onClose={() => setEditing(null)}
        onDone={async () => { setEditing(null); await refresh(); }} />}
    </>
  );
}

function History({ s, names }: { s: Segment; names: Record<string, string> }) {
  return (
    <details className="lw-transcript-history">
      <summary className="lw-meta">{`Corrected ${s.revisions!.length === 1 ? 'once' : `${s.revisions!.length} times`} — as heard: “${s.raw_text}”`}</summary>
      <ol>
        {s.revisions!.map((r) => (
          <li key={r.id} className="lw-meta">
            {r.kind === 'text' ? `Text: “${r.text}”` : `Speaker: ${names[r.speaker_participant_id ?? ''] ?? 'Unattributed'}`}
            {` — ${r.by ?? 'someone'}, ${when(r.at)}: ${r.reason}`}{r.source === 'vocabulary' ? ' (matter vocabulary)' : ''}
          </li>
        ))}
      </ol>
    </details>
  );
}

function CorrectDialog({ base, s, participants, onClose, onDone }: {
  base: string; s: Segment; participants: TranscriptData['participants']; onClose: () => void; onDone: () => void;
}) {
  const [text, setText] = useState(s.text);
  const [speaker, setSpeaker] = useState(s.speaker_participant_id ?? '');
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const changed = text.trim() !== s.text || speaker !== (s.speaker_participant_id ?? '');
  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setError('');
    try {
      await api.post(`${base}/segments/${encodeURIComponent(s.id)}/revisions`, {
        text: text.trim() !== s.text ? text.trim() : undefined,
        speaker_participant_id: speaker !== (s.speaker_participant_id ?? '') && speaker ? speaker : undefined, reason: reason.trim(),
      });
      onDone();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The correction could not be saved.');
      setBusy(false);
    }
  };
  return (
    <Dialog title="Correct this segment" onClose={onClose} footer={
      <>
        <button type="button" className="lw-btn lw-btn-ghost" onClick={onClose}>Cancel</button>
        <button type="submit" form="correct-form" className="lw-btn lw-btn-primary" disabled={busy || !changed || reason.trim().length < 3}>Save correction</button>
      </>}>
      <form id="correct-form" onSubmit={save} style={{ display: 'grid', gap: 12 }}>
        <p className="lw-meta" style={{ margin: 0 }}>{`As heard: “${s.raw_text ?? s.text}”. The original stays in the record; your correction is added with your name and reason.`}</p>
        <Field label="Text">{(id) => <textarea id={id} className="lw-input lw-serif" rows={3} value={text} onChange={(e) => setText(e.target.value)} />}</Field>
        <Field label="Speaker">{(id) => (
          <select id={id} className="lw-input" value={speaker} onChange={(e) => setSpeaker(e.target.value)}>
            <option value="">Unattributed</option>
            {participants.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
        )}</Field>
        <Field label="Reason" hint="For example: checked against the recording notes, or the redline.">{(id) => (
          <input id={id} className="lw-input" value={reason} onChange={(e) => setReason(e.target.value)} />
        )}</Field>
        {error && <ErrorNote>{error}</ErrorNote>}
      </form>
    </Dialog>
  );
}
