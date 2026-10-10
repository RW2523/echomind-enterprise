import React, { useState } from 'react';
import { Link } from 'react-router';
import { BookOpenText, Plus } from 'lucide-react';
import { api, ApiError } from '../api/client';
import { useDocuments } from '../api/queries';
import type { AgendaItem } from '../api/types';
import { useMatter } from '../app/matter';
import { ErrorNote, Field } from './primitives';

/** The agenda and the board pack each item relies on (S2.6). */
export function AgendaEditor({ base, agenda, canEdit, onDone }: { base: string; agenda: AgendaItem[]; canEdit: boolean; onDone: () => void }) {
  const matter = useMatter();
  const docs = useDocuments(matter.id);
  const [title, setTitle] = useState('');
  const [chosen, setChosen] = useState<string[]>([]);
  const [error, setError] = useState('');
  const add = async (e: React.FormEvent) => {
    e.preventDefault(); setError('');
    try { await api.post(`${base}/agenda`, { title: title.trim(), document_ids: chosen }); setTitle(''); setChosen([]); onDone(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The item could not be added.'); }
  };
  const remove = async (id: string) => {
    setError('');
    try { await api.del(`${base}/agenda/${encodeURIComponent(id)}`); onDone(); }
    catch (err) { setError(err instanceof ApiError ? err.message : 'The item could not be removed.'); }
  };
  return (
    <div style={{ borderTop: '1px solid var(--rule)', paddingTop: 14 }}>
      <h3 className="lw-eyebrow" style={{ margin: '0 0 6px' }}>Agenda and board pack</h3>
      {agenda.length === 0 ? <p className="lw-meta" style={{ margin: '0 0 8px' }}>No agenda yet. Minutes are filed under its items.</p> : (
        <ol className="lw-agenda" aria-label="Agenda">
          {agenda.map((a, i) => (
            <li key={a.id}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
                <span className="lw-tnum lw-meta">{i + 1}.</span><b>{a.title}</b>
                {a.documents.length > 0 && (
                  <Link className="lw-linklike lw-meta" style={{ marginLeft: 'auto' }}
                    to={`/m/${matter.number}/research?docs=${a.documents.map((d) => encodeURIComponent(d.id)).join(',')}&about=${encodeURIComponent(a.title)}`}>
                    <BookOpenText size={13} /> Research these papers</Link>
                )}
                {canEdit && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => remove(a.id)} aria-label={`Remove ${a.title}`}>Remove</button>}
              </div>
              {a.documents.length > 0 && <div className="lw-meta">{a.documents.map((d) => d.title).join(' · ')}</div>}
            </li>
          ))}
        </ol>
      )}
      {canEdit && (
        <form onSubmit={add} style={{ display: 'grid', gap: 8, marginTop: 8 }}>
          <Field label="Agenda item">{(id) => <input id={id} className="lw-input" value={title} onChange={(e) => setTitle(e.target.value)} />}</Field>
          {(docs.data?.length ?? 0) > 0 && (
            <fieldset className="lw-agenda-docs">
              <legend className="lw-meta">Board papers for this item</legend>
              {docs.data!.map((d) => (
                <label key={d.id} className="lw-meta">
                  <input type="checkbox" checked={chosen.includes(d.id)}
                    onChange={(e) => setChosen((c) => (e.target.checked ? [...c, d.id] : c.filter((x) => x !== d.id)))} /> {d.title}
                </label>
              ))}
            </fieldset>
          )}
          <div><button type="submit" className="lw-btn lw-btn-sm" disabled={!title.trim()}><Plus />Add to the agenda</button></div>
        </form>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}
    </div>
  );
}
