import React, { useState } from 'react';
import { Link } from 'react-router';
import { ListCollapse } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { useDocSummaries, useDocuments, useSummaries } from '../api/queries';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field, Skeleton, StatusPill } from '../ui/primitives';
import { reviewState } from '../ui/review';

const day = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/** A matter's summaries (S4.2, D-055): each document's key points, every one quoting the document with its page and
 *  line, clause or transcript time. A second person approves before a summary is relied on; the original is unchanged. */
export default function MatterSummaries() {
  const matter = useMatter();
  const q = useSummaries(matter.id);
  const docs = useDocuments(matter.id);
  const qc = useQueryClient();
  const [docId, setDocId] = useState('');
  const [started, setStarted] = useState<string | null>(null);
  const [error, setError] = useState('');
  const job = useDocSummaries(matter.id, started);
  const ready = (docs.data ?? []).filter((d) => d.current_version?.index_status === 'indexed');
  const running = job.data?.job && ['queued', 'running'].includes(job.data.job.status);
  React.useEffect(() => {
    if (started && job.data?.job?.status === 'done') { void qc.invalidateQueries({ queryKey: ['matter', matter.id, 'summaries'] }); setStarted(null); }
    if (started && job.data?.job?.status === 'failed') { setError('The summary could not be made. Try again, or ask your administrator.'); setStarted(null); }
  }, [job.data, started, matter.id, qc]);
  const summarise = async () => {
    setError('');
    try {
      await api.post(`/api/legal/matters/${encodeURIComponent(matter.id)}/documents/${encodeURIComponent(docId)}/summaries`);
      setStarted(docId);
    } catch (e) { setError(e instanceof ApiError ? e.message : 'The summary could not be started.'); }
  };
  return (
    <>
      <Toolbar crumb={matter.number} title="Summaries" />
      <div className="lw-content">
        <p className="lw-meta" style={{ margin: 0 }}>Each point quotes the document and says where: page and line, the clause, or the time in a transcript. A point whose words are not found is marked. Summaries are drafts until a second person approves them; the original is never changed. A summary can be translated from its page when the firm has turned translation on.</p>
        {q.data?.can.summarise && (
          <section className="lw-paper lw-sheet" aria-label="Summarise a document">
            <div style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
              <Field label="Document">{(id) => (
                <select id={id} className="lw-input" value={docId} onChange={(e) => setDocId(e.target.value)}>
                  <option value="">Choose a document</option>
                  {ready.map((d) => <option key={d.id} value={d.id}>{d.title}</option>)}
                </select>)}</Field>
              <button type="button" className="lw-btn lw-btn-primary" disabled={!docId || !!running} onClick={summarise}><ListCollapse />{running ? 'Summarising…' : 'Summarise'}</button>
            </div>
            {running && <p className="lw-meta" role="status" style={{ margin: '8px 0 0' }}>Reading the document section by section. This can take a minute for a long document.</p>}
          </section>
        )}
        {error && <ErrorNote>{error}</ErrorNote>}
        {q.isPending ? <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div>
          : q.isError ? <ErrorNote>The summaries could not be loaded.</ErrorNote>
            : q.data.summaries.length === 0 ? <EmptyState title="No summaries yet">{q.data.can.summarise ? 'Choose a document above.' : 'A lawyer on the matter makes them.'}</EmptyState>
              : (
                <ul className="lw-check-lines lw-paper lw-sheet" aria-label="Summaries">
                  {q.data.summaries.map((s) => {
                    const st = reviewState(s);
                    return (
                      <li key={s.id} aria-label={s.content.title}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                          <Link to={`/m/${encodeURIComponent(matter.number)}/summaries/${encodeURIComponent(s.id)}`}><b>{s.content.title}</b></Link>
                          <StatusPill tone={st.tone}>{st.label}</StatusPill>
                        </div>
                        <span className="lw-meta">{`${s.content.points.length} points · version ${s.version} · ${day(s.created_at)}`}{s.content.unverified ? ` · ${s.content.unverified} not verified` : ''}</span>
                      </li>
                    );
                  })}
                </ul>
              )}
      </div>
    </>
  );
}
