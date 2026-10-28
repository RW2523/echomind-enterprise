import React, { useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router';
import { ArrowLeft, Download } from 'lucide-react';
import { postStream } from '../api/client';
import { useDocument } from '../api/queries';
import { useMatter } from '../app/matter';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Field } from '../ui/primitives';
import { LANGUAGES, RTL, TranslateIcon, TranslationChecks, TranslationLabel } from '../ui/translation';

interface Row { i: number; source: string; text: string; withheld: boolean; figures_missing: string[] }
interface Run { label: string; language: { code: string; name: string } | null; total: number; rows: Row[]; next: number | null; remaining: number;
  phase: 'idle' | 'running' | 'done' | 'error' | 'revoked' }
const EMPTY: Run = { label: '', language: null, total: 0, rows: [], next: null, remaining: 0, phase: 'idle' };

/** A document translated by the local model (OPT-02, D-070), paragraph by paragraph beside the original. Nothing is
 *  stored; the label says it is a machine translation no one has reviewed, and figures missing from it are flagged. */
export default function DocumentTranslate() {
  const matter = useMatter();
  const me = useSession();
  const { docId = '' } = useParams();
  const doc = useDocument(matter.id, docId);
  const [lang, setLang] = useState('');
  const [run, setRun] = useState<Run>(EMPTY);
  const ctrl = useRef<AbortController | null>(null);
  useEffect(() => () => ctrl.current?.abort(), []);
  const title = doc.data?.title ?? 'Document';
  const back = <Link className="lw-btn lw-btn-sm lw-btn-ghost" to={`/m/${matter.number}/documents/${encodeURIComponent(docId)}/read`}><ArrowLeft />Back to the document</Link>;

  const translate = async (from?: number) => {
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setRun((r) => (from ? { ...r, next: null, phase: 'running' } : { ...EMPTY, phase: 'running' }));
    try {
      await postStream(`/api/legal/matters/${encodeURIComponent(matter.id)}/documents/${encodeURIComponent(docId)}/translate`,
        from ? { language: lang, from } : { language: lang }, (ev) => {
          if (ev.type === 'start') setRun((r) => ({ ...r, label: ev.label, language: ev.language, total: ev.paragraphs }));
          else if (ev.type === 'paragraph') setRun((r) => ({ ...r, rows: [...r.rows, ev as Row] }));
          else if (ev.type === 'limit') setRun((r) => ({ ...r, next: ev.next, remaining: ev.remaining }));
          else if (ev.type === 'done') setRun((r) => ({ ...r, phase: 'done' }));
          else if (ev.type === 'scope_revoked') setRun((r) => ({ ...r, rows: [], phase: 'revoked' }));
          else if (ev.type === 'error') setRun((r) => ({ ...r, phase: 'error' }));
        }, c.signal);
    } catch {
      if (!c.signal.aborted) setRun((r) => ({ ...r, phase: 'error' }));
    }
  };
  const download = () => {
    const body = [`${title} — ${run.language?.name ?? ''}`, run.label, '',
      ...run.rows.flatMap((r) => [`[${r.i + 1}] ${r.source}`, `[${r.i + 1}] ${r.text}`, ''])].join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([body], { type: 'text/plain;charset=utf-8' }));
    a.download = `${title.replace(/\.[a-z0-9]+$/i, '')} — ${run.language?.name ?? 'translation'} (machine translation).txt`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  if (!me.features?.translation) {
    return (
      <>
        <Toolbar crumb={`${matter.number} · Documents`} title="Translate">{back}</Toolbar>
        <div className="lw-content"><EmptyState title="Translation is off on this server">The firm can turn on machine translation by this server's own model.</EmptyState></div>
      </>
    );
  }
  return (
    <>
      <Toolbar crumb={`${matter.number} · Documents`} title={`Translate — ${title}`}>{back}</Toolbar>
      <div className="lw-content">
        <section className="lw-paper lw-sheet" aria-label="Translate">
          <div style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
            <Field label="Into">{(id) => (
              <select id={id} className="lw-input" value={lang} onChange={(e) => setLang(e.target.value)}>
                <option value="">Choose a language</option>
                {LANGUAGES.map(([code, name]) => <option key={code} value={code}>{name}</option>)}
              </select>)}</Field>
            <button type="button" className="lw-btn lw-btn-primary" disabled={!lang || run.phase === 'running'} onClick={() => translate()}>
              <TranslateIcon />{run.phase === 'running' ? 'Translating…' : 'Translate'}
            </button>
            {run.rows.length > 0 && run.phase !== 'running' && (
              <button type="button" className="lw-btn lw-btn-ghost" onClick={download}><Download />Download (.txt)</button>
            )}
          </div>
          <p className="lw-meta" style={{ margin: '8px 0 0' }}>Translated by the model configured for this server, a few paragraphs at a time; the translation is not kept. Every figure of the original is looked for in the translation.</p>
        </section>
        {run.label && <TranslationLabel text={run.label} />}
        {run.phase === 'running' && <p className="lw-meta" role="status" style={{ margin: 0 }}>{`Translating into ${run.language?.name ?? '…'}: ${run.rows.length} of ${run.total} paragraphs.`}</p>}
        {run.phase === 'error' && <ErrorNote>The translation could not be completed. Try again.</ErrorNote>}
        {run.phase === 'revoked' && <ErrorNote>Your access to this matter changed, so the translation stopped.</ErrorNote>}
        {run.rows.length > 0 && (
          <ol className="lw-translate-rows" aria-label="Original and translation">
            {run.rows.map((r) => (
              <li key={r.i}>
                <p className="lw-serif">{r.source}</p>
                <div style={{ display: 'grid', gap: 6, alignContent: 'start' }}>
                  <p className="lw-serif" lang={run.language?.code} dir={RTL.has(run.language?.code ?? '') ? 'rtl' : undefined}>{r.text}</p>
                  <TranslationChecks figures={r.figures_missing} withheld={r.withheld} />
                </div>
              </li>
            ))}
          </ol>
        )}
        {run.next != null && run.phase === 'done' && (
          <div><button type="button" className="lw-btn" onClick={() => translate(run.next!)}>{`Continue (${run.remaining} ${run.remaining === 1 ? 'paragraph' : 'paragraphs'} left)`}</button></div>
        )}
      </div>
    </>
  );
}
