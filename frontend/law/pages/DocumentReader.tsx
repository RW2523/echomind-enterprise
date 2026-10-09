import React, { useEffect, useMemo, useRef } from 'react';
import { Link, useParams, useSearchParams } from 'react-router';
import { ArrowLeft } from 'lucide-react';
import { ApiError } from '../api/client';
import { useDocument, useReader } from '../api/queries';
import type { ReaderData, ReaderMark, ReaderPage } from '../api/types';
import { useMatter } from '../app/matter';
import Toolbar from '../ui/Toolbar';
import { ErrorNote, Skeleton, StatusPill } from '../ui/primitives';

/**
 * The document reader (S2.1): a version's text as EchoMind read it, page by page with line numbers,
 * opened at the quote a citation points to. Revisions, comments, text-recognition confidence and
 * text addressed to an AI are marked in place; hidden text that was left out is listed beside it.
 */
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
export const longDate = (iso?: string | null) => {
  if (!iso) return '';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
};
const STATUS: Record<string, string> = { draft: 'Draft', final: 'Final', executed: 'Executed', received: 'Received' };
const HIDDEN: Record<string, string> = {
  white: 'White text', tiny: 'Tiny text', invisible: 'Invisible text', off_page: 'Outside the page', hidden_run: 'Hidden in Word',
};

type Range = { start: number; end: number; kind: ReaderMark['kind'] | 'focus'; meta?: Record<string, any> };

function parseRange(at: string | null): [number, number] | null {
  const m = /^(\d+)-(\d+)$/.exec(at ?? '');
  return m && Number(m[2]) > Number(m[1]) ? [Number(m[1]), Number(m[2])] : null;
}

export default function DocumentReader() {
  const { docId = '' } = useParams();
  const [params] = useSearchParams();
  const matter = useMatter();
  const doc = useDocument(matter.id, docId);
  const versionId = params.get('v') || doc.data?.current_version?.id || null;
  const reader = useReader(matter.id, docId, versionId);
  const focus = parseRange(params.get('at'));
  const title = doc.data?.title ?? 'Document';
  return (
    <>
      <Toolbar crumb={`${matter.number} · Documents`} title={title}>
        <Link className="lw-btn lw-btn-sm lw-btn-ghost" to={`/m/${matter.number}/documents`}><ArrowLeft />Documents</Link>
      </Toolbar>
      {reader.isPending ? <div className="lw-paper" style={{ padding: 24 }}><Skeleton /><Skeleton /><Skeleton /></div>
        : reader.isError ? <ErrorNote>{reader.error instanceof ApiError ? reader.error.message : 'This document cannot be opened.'}</ErrorNote>
        : <ReaderBody data={reader.data} focus={focus} />}
    </>
  );
}

function ReaderBody({ data, focus }: { data: ReaderData; focus: [number, number] | null }) {
  const sheet = useRef<HTMLElement>(null);
  useEffect(() => {
    sheet.current?.querySelector('[data-focus]')?.scrollIntoView?.({ block: 'center' });
  }, [data, focus?.[0], focus?.[1]]);
  const jump = (offset: number) => {
    const lines = Array.from(sheet.current?.querySelectorAll<HTMLElement>('[data-a]') ?? []);
    const line = lines.find((l) => Number(l.dataset.a) <= offset && offset <= Number(l.dataset.b)) ?? lines.find((l) => Number(l.dataset.a) >= offset);
    line?.scrollIntoView?.({ block: 'center' });
  };
  const v = data.version;
  return (
    <div className="lw-reader">
      <nav className="lw-paper lw-reader-outline" aria-label="Clauses">
        <h2 className="lw-eyebrow" style={{ margin: '0 0 8px' }}>Clauses</h2>
        {data.clauses.length === 0 ? <p className="lw-meta" style={{ margin: 0 }}>No numbered clauses found.</p> : (
          <ul>
            {data.clauses.map((c) => (
              <li key={`${c.number}-${c.start}`} style={{ paddingLeft: Math.min(c.level, 4) * 12 }}>
                <button type="button" className="lw-linklike" onClick={() => jump(c.start)}>
                  <span className="lw-tnum" style={{ fontWeight: 600 }}>{c.number}</span>{c.heading ? ` ${c.heading}` : ''}
                </button>
              </li>
            ))}
          </ul>
        )}
        {data.terms.length > 0 && (
          <>
            <h2 className="lw-eyebrow" style={{ margin: '16px 0 8px' }}>Defined terms</h2>
            <ul>{data.terms.map((t) => <li key={t.term}><button type="button" className="lw-linklike" onClick={() => jump(t.start)}>{t.term}</button></li>)}</ul>
          </>
        )}
      </nav>
      <article ref={sheet} className="lw-paper lw-reader-sheet" aria-label="Document text">
        <div className="lw-meta" style={{ marginBottom: 12 }}>
          v{v.number} · {v.filename}{v.status ? <> · <StatusPill tone={v.status === 'executed' ? 'verified' : 'neutral'}>{STATUS[v.status] ?? v.status}</StatusPill></> : null}
        </div>
        <Text data={data} focus={focus} />
      </article>
      <Found data={data} />
    </div>
  );
}

function Text({ data, focus }: { data: ReaderData; focus: [number, number] | null }) {
  const ranges = useMemo<Range[]>(() => {
    const out: Range[] = data.marks.filter((m) => m.start != null && m.end != null && m.kind !== 'tracked_delete' && m.kind !== 'hidden_text')
      .map((m) => ({ start: m.start!, end: m.end!, kind: m.kind, meta: m.meta }));
    if (focus) out.push({ start: focus[0], end: focus[1], kind: 'focus' });
    return out;
  }, [data, focus]);
  const deletions = useMemo(() => data.marks.filter((m) => m.kind === 'tracked_delete' && m.start != null), [data]);
  const pages: ReaderPage[] = data.pages.length ? data.pages : [{ number: 0, start: 0, end: data.text.length, source: 'text' }];
  return (
    <>
      {pages.map((p) => {
        let n = 0;
        let pos = p.start;
        const lines = data.text.slice(p.start, p.end).split('\n').map((line) => {
          const a = pos;
          pos += line.length + 1;
          if (!line.trim()) return <div key={a} className="lw-reader-gap" aria-hidden="true" />;
          n += 1;
          const label = p.number ? `${p.number}.${n}` : `${n}`;
          return (
            <div key={a} className="lw-reader-line" data-line={label} data-a={a} data-b={a + line.length}>
              <span className="lw-reader-no" aria-hidden="true">{n}</span>
              <span className="lw-reader-text">{segments(data.text, a, a + line.length, ranges, deletions)}</span>
            </div>
          );
        });
        return (
          <section key={p.number} className="lw-reader-page" aria-label={p.number ? `Page ${p.number}` : 'Text'}>
            {p.number > 0 && (
              <div className={`lw-reader-page-h${p.source === 'ocr' ? ' lw-reader-ocr' : ''}`}>
                {p.source === 'ocr'
                  ? `Page ${p.number} · read by text recognition, ${Math.round(p.confidence ?? 0)}% confidence — check against the scan`
                  : p.source === 'image' ? `Page ${p.number} · a scan that was not read` : `Page ${p.number}`}
              </div>
            )}
            {lines}
          </section>
        );
      })}
    </>
  );
}

function segments(text: string, a: number, b: number, ranges: Range[], deletions: ReaderMark[]): React.ReactNode[] {
  const inLine = ranges.filter((r) => r.start < b && r.end > a);
  const points = new Set<number>([a, b]);
  for (const r of inLine) { points.add(Math.max(a, r.start)); points.add(Math.min(b, r.end)); }
  for (const d of deletions) if (d.start! >= a && d.start! <= b) points.add(d.start!);
  const cuts = [...points].sort((x, y) => x - y);
  const out: React.ReactNode[] = [];
  let focusRun: React.ReactNode[] | null = null;
  const flush = () => { if (focusRun) { out.push(<mark key={`f${out.length}`} data-focus="" className="lw-reader-focus">{focusRun}</mark>); focusRun = null; } };
  for (let i = 0; i < cuts.length; i++) {
    const x = cuts[i];
    for (const d of deletions) {
      if (d.start === x && (x < b || x === b)) {
        const del = <del key={`d${x}`} className="lw-reader-del" title={`Deleted by ${d.meta.author ?? 'someone'}, ${longDate(d.meta.date)}`}>{d.meta.text}</del>;
        (focusRun ?? out).push(del);
      }
    }
    const y = cuts[i + 1];
    if (y === undefined || y <= x) continue;
    const covering = inLine.filter((r) => r.start <= x && r.end >= y);
    let node: React.ReactNode = text.slice(x, y);
    for (const r of covering.filter((r) => r.kind === 'ocr_low_confidence')) node = <span key="l" data-low-confidence="" className="lw-reader-low" title={`Recognised with ${r.meta?.confidence}% confidence`}>{node}</span>;
    for (const r of covering.filter((r) => r.kind === 'tracked_insert')) node = <ins key="i" className="lw-reader-ins" title={`Inserted by ${r.meta?.author ?? 'someone'}, ${longDate(r.meta?.date)}`}>{node}</ins>;
    for (const r of covering.filter((r) => r.kind === 'instructions')) node = <span key="a" className="lw-reader-ai" title="Addressed to an AI assistant — withheld from answers">{node}</span>;
    for (const r of covering.filter((r) => r.kind === 'comment')) node = <span key="c" className="lw-reader-comment" title={`${r.meta?.author}: ${r.meta?.text}`}>{node}</span>;
    const isFocus = covering.some((r) => r.kind === 'focus');
    if (isFocus) { (focusRun ??= []).push(<React.Fragment key={x}>{node}</React.Fragment>); }
    else { flush(); out.push(<React.Fragment key={x}>{node}</React.Fragment>); }
  }
  flush();
  return out;
}

function Found({ data }: { data: ReaderData }) {
  const hidden = data.marks.filter((m) => m.kind === 'hidden_text');
  const ai = data.marks.filter((m) => m.kind === 'instructions');
  const comments = data.marks.filter((m) => m.kind === 'comment');
  const changes = data.marks.filter((m) => m.kind === 'tracked_insert' || m.kind === 'tracked_delete');
  const ocr = data.pages.filter((p) => p.source === 'ocr');
  const nothing = !hidden.length && !ai.length && !comments.length && !changes.length && !ocr.length;
  return (
    <aside className="lw-paper lw-reader-found" aria-label="Found while reading">
      <h2 className="lw-eyebrow" style={{ margin: '0 0 8px' }}>Found while reading</h2>
      {nothing && <p className="lw-meta" style={{ margin: 0 }}>Nothing hidden, no revisions or comments.</p>}
      {hidden.length > 0 && (
        <section aria-label="Hidden text">
          <h3 className="lw-reader-h3">Hidden text, left out</h3>
          <ul>{hidden.map((m, i) => (
            <li key={i} className="lw-reader-item lw-reader-item-conflict">
              <div className="lw-meta">{HIDDEN[m.meta.reason] ?? 'Hidden'}{m.page ? ` · page ${m.page}` : ''}{m.meta.instructions ? ' · addressed to an AI assistant' : ''}</div>
              <div className="lw-reader-quote">{m.meta.text}</div>
            </li>))}</ul>
        </section>
      )}
      {ai.length > 0 && (
        <section aria-label="Addressed to an AI">
          <h3 className="lw-reader-h3">Addressed to an AI assistant</h3>
          <p className="lw-meta" style={{ margin: '0 0 6px' }}>Withheld from answers; marked in the text.</p>
          <ul>{ai.map((m, i) => <li key={i} className="lw-reader-item lw-reader-item-attention"><div className="lw-reader-quote">{data.text.slice(m.start!, m.end!)}</div></li>)}</ul>
        </section>
      )}
      {comments.length > 0 && (
        <section aria-label="Comments">
          <h3 className="lw-reader-h3">Comments</h3>
          <ul>{comments.map((m, i) => (
            <li key={i} className="lw-reader-item">
              <div className="lw-meta">{m.meta.author} · {longDate(m.meta.date)}</div>
              <div>{m.meta.text}</div>
              <div className="lw-meta lw-ellipsis">On “{data.text.slice(m.start!, m.end!)}”</div>
            </li>))}</ul>
        </section>
      )}
      {changes.length > 0 && (
        <section aria-label="Tracked changes">
          <h3 className="lw-reader-h3">Tracked changes, read as accepted</h3>
          <ul>{changes.map((m, i) => (
            <li key={i} className="lw-reader-item">
              <div className="lw-meta">{m.kind === 'tracked_insert' ? 'Inserted' : 'Deleted'} by {m.meta.author ?? 'someone'} · {longDate(m.meta.date)}</div>
              <div className="lw-reader-quote">{m.kind === 'tracked_insert' ? data.text.slice(m.start!, m.end!) : m.meta.text}</div>
            </li>))}</ul>
        </section>
      )}
      {ocr.length > 0 && (
        <section aria-label="Text recognition">
          <h3 className="lw-reader-h3">Text recognition</h3>
          <p className="lw-meta" style={{ margin: 0 }}>{ocr.length} page{ocr.length === 1 ? '' : 's'} read from the scan ({data.engines.ocr ?? 'OCR'}). Words recognised with low confidence are underlined.</p>
        </section>
      )}
    </aside>
  );
}
