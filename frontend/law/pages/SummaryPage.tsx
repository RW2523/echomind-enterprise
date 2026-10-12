import React, { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router';
import { ArrowDown, ArrowLeft, ArrowUp, FileOutput, Pencil, Trash2 } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { artifactPath, summariesPath, useArtifact } from '../api/queries';
import type { SummaryContent, SummaryPoint, SummaryVersion } from '../api/types';
import { useMatter } from '../app/matter';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { ErrorNote, Field, Skeleton, StatusPill } from '../ui/primitives';
import { ApprovalDialog, ReviewBanner, ReviewDecisions } from '../ui/review';
import { citeLabel } from '../ui/summary';
import { LANGUAGES, RTL, TranslateIcon, TranslationChecks, TranslationLabel } from '../ui/translation';

const day = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/** One summary (S4.2, D-055): key points in the document's order, each with the words it rests on and where they
 *  are. A reviewer rewords, reorders or removes points as a new version; a second person approves; it is filed as a
 *  matter document. The original is never changed. */
export default function SummaryPage() {
  const { artifactId = '' } = useParams();
  const matter = useMatter();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const view = useArtifact<SummaryContent>(matter.id, artifactId);
  const [editing, setEditing] = useState<SummaryPoint[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [moved, setMoved] = useState<string | null>(null);
  const me = useSession();
  const [lang, setLang] = useState('');
  const [translated, setTranslated] = useState<{ label: string; code: string; points: Record<string, { translation: string; figures_missing: string[] }> } | null>(null);
  const [translating, setTranslating] = useState(false);
  const refresh = () => Promise.all([qc.invalidateQueries({ queryKey: ['matter', matter.id, 'artifact'] }),
    qc.invalidateQueries({ queryKey: ['matter', matter.id, 'summaries'] }), qc.invalidateQueries({ queryKey: ['reviews'] })]);
  const back = <Link className="lw-btn lw-btn-sm lw-btn-ghost" to={`/m/${matter.number}/summaries`}><ArrowLeft />Summaries</Link>;
  if (view.isPending) return <><Toolbar crumb={`${matter.number} · Summaries`} title="Summary">{back}</Toolbar><div className="lw-content"><Skeleton height={200} /></div></>;
  if (view.isError) return <><Toolbar crumb={`${matter.number} · Summaries`} title="Summary">{back}</Toolbar><div className="lw-content"><ErrorNote>This summary is not available.</ErrorNote></div></>;
  const { artifact: v, can, versions } = view.data;
  const c = v.content;
  const newest = versions[0]?.id === v.id;
  const reader = (p: SummaryPoint) => (p.start != null && p.end != null
    ? `/m/${matter.number}/documents/${encodeURIComponent(c.document.id)}/read?v=${encodeURIComponent(c.document.version_id)}&at=${p.start}-${p.end}` : null);
  const save = async () => {
    if (!editing) return;
    setBusy(true); setError('');
    try {
      const r = await api.post<{ summary: SummaryVersion }>(`${summariesPath(matter.id)}/${encodeURIComponent(v.id)}/edit`,
        { points: editing.map((p) => ({ id: p.id, text: p.text })) });
      setEditing(null);
      await refresh();
      navigate(`/m/${matter.number}/summaries/${encodeURIComponent(r.summary.id)}`, { replace: true });
    } catch (e) { setError(e instanceof ApiError ? e.message : 'The edit could not be saved.'); } finally { setBusy(false); }
  };
  const file = async () => {
    setBusy(true); setError('');
    try { await api.post(`${artifactPath(matter.id, v.id)}/file`); await refresh(); }
    catch (e) { setError(e instanceof ApiError ? e.message : 'The summary could not be filed.'); } finally { setBusy(false); }
  };
  const translate = async () => {
    setTranslating(true); setError('');
    try {
      const r = await api.post<{ label: string; language: { code: string }; points: { id: string; translation: string; figures_missing: string[] }[] }>(
        `${summariesPath(matter.id)}/${encodeURIComponent(v.id)}/translate`, { language: lang });
      setTranslated({ label: r.label, code: r.language.code, points: Object.fromEntries(r.points.map((x) => [x.id, x])) });
    } catch (e) { setError(e instanceof ApiError ? e.message : 'The translation could not be made.'); } finally { setTranslating(false); }
  };
  const move = (i: number, by: number) => setEditing((pts) => {
    if (!pts) return pts;
    const next = [...pts];
    const [x] = next.splice(i, 1);
    next.splice(i + by, 0, x);
    setMoved(`${x.text.slice(0, 40)} moved ${by < 0 ? 'up' : 'down'}`);
    return next;
  });
  return (
    <>
      <Toolbar crumb={`${matter.number} · Summaries`} title="Summary">{back}</Toolbar>
      <div className="lw-content" style={{ maxWidth: 960 }}>
        <article className="lw-paper lw-sheet" aria-label="Summary" aria-busy={busy}>
          <p className="lw-eyebrow" style={{ margin: 0 }}>{c.transcript ? 'Transcript' : 'Document'}</p>
          <h2 className="lw-h2 lw-serif" style={{ margin: '2px 0 6px' }}>{c.title}</h2>
          <ReviewBanner v={v} made={`Version ${v.version} · ${v.origin === 'model' ? 'made by the model for' : 'edited by'} ${v.created_by_name ?? 'someone'}, ${day(v.created_at)}`} />
          <p className="lw-meta" style={{ margin: '0 0 10px' }}>
            {`Of `}<Link to={`/m/${matter.number}/documents/${encodeURIComponent(c.document.id)}/read?v=${encodeURIComponent(c.document.version_id)}`}>{c.document.title}</Link>{`, version ${c.document.version_number} — the original is unchanged.`}
            {c.unverified ? ` ${c.unverified} point(s) not verified.` : ''}{c.figures_unsupported ? ` ${c.figures_unsupported} point(s) state a figure their quote does not contain.` : ''}
          </p>
          {!editing && me.features?.translation && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap', margin: '0 0 10px' }}>
              <Field label="Translate into">{(id) => (
                <select id={id} className="lw-input" value={lang} onChange={(e) => setLang(e.target.value)}>
                  <option value="">Choose a language</option>
                  {LANGUAGES.map(([code, name]) => <option key={code} value={code}>{name}</option>)}
                </select>)}</Field>
              <button type="button" className="lw-btn lw-btn-sm" disabled={!lang || translating} onClick={translate}><TranslateIcon />{translating ? 'Translating…' : 'Translate'}</button>
            </div>
          )}
          {!editing && translated && <TranslationLabel text={translated.label} />}
          {editing ? (
            <ol className="lw-summary-points" aria-label="Edit the points">
              {editing.map((p, i) => (
                <li key={p.id}>
                  <label className="lw-sr" htmlFor={`pt-${p.id}`}>{`Point ${i + 1}`}</label>
                  <textarea id={`pt-${p.id}`} className="lw-input" rows={2} maxLength={600} value={p.text}
                    onChange={(e) => setEditing(editing.map((x) => (x.id === p.id ? { ...x, text: e.target.value } : x)))} />
                  <div className="lw-summary-edit">
                    <span className="lw-meta">{citeLabel(p.cite)}</span>
                    <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" aria-label={`Move point ${i + 1} up`} disabled={i === 0} onClick={() => move(i, -1)}><ArrowUp /></button>
                    <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" aria-label={`Move point ${i + 1} down`} disabled={i === editing.length - 1} onClick={() => move(i, 1)}><ArrowDown /></button>
                    <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" aria-label={`Remove point ${i + 1}`} disabled={editing.length === 1}
                      onClick={() => setEditing(editing.filter((x) => x.id !== p.id))}><Trash2 /></button>
                  </div>
                </li>
              ))}
            </ol>
          ) : (
            <ol className="lw-summary-points" aria-label="Key points">
              {c.points.map((p) => {
                const href = reader(p);
                return (
                  <li key={p.id} aria-label={`${p.text} (${citeLabel(p.cite)})`} data-verification={p.verification}>
                    <p className="lw-serif" style={{ margin: 0 }}>{p.text}</p>
                    <div className="lw-summary-cite">
                      {href ? <Link to={href} className="lw-cite-chip">{citeLabel(p.cite)}</Link> : <StatusPill tone="conflict">Not verified</StatusPill>}
                      {p.origin === 'person' && <StatusPill tone="neutral">Edited by a reviewer</StatusPill>}
                      {p.figures_unsupported.length > 0 && <StatusPill tone="attention">{`Check: ${p.figures_unsupported.join(', ')} not in the quote`}</StatusPill>}
                    </div>
                    {p.quote && <blockquote className="lw-summary-quote">“{p.quote}”</blockquote>}
                    {translated?.points[p.id] && (
                      <div className="lw-translation">
                        <p className="lw-serif" lang={translated.code} dir={RTL.has(translated.code) ? 'rtl' : undefined} style={{ margin: 0 }}>{translated.points[p.id].translation}</p>
                        <TranslationChecks figures={translated.points[p.id].figures_missing} />
                      </div>
                    )}
                  </li>
                );
              })}
            </ol>
          )}
          <span className="lw-sr" aria-live="polite">{moved ?? ''}</span>
          {error && <ErrorNote>{error}</ErrorNote>}
          <div className="lw-minutes-actions">
            {editing ? (
              <>
                <button type="button" className="lw-btn lw-btn-sm lw-btn-primary" disabled={busy} onClick={save}>Save as a new version</button>
                <button type="button" className="lw-btn lw-btn-sm" onClick={() => setEditing(null)}>Cancel</button>
              </>
            ) : (
              <>
                {newest && v.status !== 'approved' && can.file && <button type="button" className="lw-btn lw-btn-sm" onClick={() => setEditing(c.points.map((p) => ({ ...p })))}><Pencil />Edit points</button>}
                {v.status === 'approved' && (v.document_id
                  ? <Link className="lw-btn lw-btn-sm" to={`/m/${matter.number}/documents/${encodeURIComponent(v.document_id)}/read`}><FileOutput />Open the filed summary</Link>
                  : can.file && <button type="button" className="lw-btn lw-btn-sm" disabled={busy} onClick={file}><FileOutput />{busy ? 'Filing…' : 'File as a matter document'}</button>)}
              </>
            )}
          </div>
          {!editing && <ApprovalDialog v={v} what="this summary" endpoint={`${artifactPath(matter.id, v.id)}/review`} canApprove={can.approve || can.author}
            isAuthor={can.author} onDone={refresh} />}
          <ReviewDecisions decisions={v.decisions} />
          {!me.features?.translation && <p className="lw-meta" style={{ marginTop: 10 }}>Translation: not available. Translation is off on this server.</p>}
          {versions.length > 1 && <p className="lw-meta" style={{ marginTop: 4 }}>{`Versions: ${versions.map((x) => `${x.version} (${x.status})`).join(' · ')}`}</p>}
        </article>
      </div>
    </>
  );
}
