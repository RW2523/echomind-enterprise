/**
 * The right-hand work surface (docs/echomind-legal/02): opens beside the conversation, never instead of it. A citation is
 * fetched from the server each time it is opened (re-authorized and re-verified there); a comparison side shows the
 * passage the card quotes. "Open full document" goes to the document reader at the exact characters.
 */
import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { AlertTriangle, BadgeCheck, CircleDashed, FileText, Plus, X } from 'lucide-react';
import { ApiError } from '../api/client';
import { counselApi } from './api';
import { usePanel, type PanelContent } from './context';
import { isWorkspacePath } from './validate';
import type { CitationDetail, ComparisonSide } from './types';

function Highlighted({ text, start, end }: { text: string; start: number; end: number }) {
  const a = Math.max(0, Math.min(start, text.length));
  const b = Math.max(a, Math.min(end, text.length));
  return <>{text.slice(0, a)}<mark>{text.slice(a, b)}</mark>{text.slice(b)}</>;
}

const statusWord = (s?: string | null) => (s === 'executed' ? 'signed' : s || '');

function CitationBody({ id, onUse }: { id: string; onUse?: () => void }) {
  const q = useQuery({ queryKey: ['counsel', 'citation', id], retry: false, staleTime: 0, gcTime: 0, queryFn: () => counselApi.citation(id) });
  if (q.isPending) return <p className="qc-meta" aria-busy="true">Opening the source…</p>;
  if (q.isError) {
    // Say which of the three it is (review S1): the source changed, it is gone or closed to you, or it did not load.
    const status = q.error instanceof ApiError ? q.error.status : 0;
    if (status === 409) return <p className="qc-error" role="alert">{q.error.message}</p>;
    if (status === 403 || status === 404 || status === 410) {
      return <p className="qc-error" role="alert">This source is no longer available to you: the document may have been removed, or your access to it changed.</p>;
    }
    return (
      <div role="alert" style={{ display: 'grid', gap: 8, justifyItems: 'start' }}>
        <p className="qc-error" style={{ margin: 0 }}>The source could not be loaded. Check the connection and try again.</p>
        <button type="button" className="qc-btn" onClick={() => void q.refetch()}>Try again</button>
      </div>
    );
  }
  const c: CitationDetail = q.data.citation;
  return (
    <>
      <p className="qc-state" data-tone={c.verified ? 'verified' : 'attention'}>
        {c.verified ? <BadgeCheck aria-hidden /> : <CircleDashed aria-hidden />}
        {c.verified ? 'Verified quote: the cited words are in this passage' : 'Not verified: the answer points here, but no quote was matched'}
      </p>
      {c.context ? (
        <blockquote className="qc-panel-quote" aria-label="Cited passage">
          {c.context.truncated_before && '…'}
          <Highlighted text={c.context.text} start={c.context.highlight_start} end={c.context.highlight_end} />
          {c.context.truncated_after && '…'}
        </blockquote>
      ) : <p className="qc-meta">No exact passage is stored for this source.</p>}
      <dl className="qc-facts">
        <dt>Document</dt><dd>{c.document.title}</dd>
        <dt>Version</dt><dd>{c.version.number} · {statusWord(c.version.status)} · {c.version.filename}
          {c.version.uploaded_at ? <>, added <time dateTime={c.version.uploaded_at}>{new Date(c.version.uploaded_at).toLocaleDateString([], { dateStyle: 'medium' })}</time></> : null}</dd>
        {c.page ? <><dt>Page</dt><dd>{c.page}{c.page_end && c.page_end !== c.page ? `–${c.page_end}` : ''}{c.line ? `, line ${c.line}` : ''}</dd></> : null}
        {c.clause ? <><dt>Clause</dt><dd>{c.clause}</dd></> : null}
        <dt>File hash</dt><dd className="qc-mono" title={c.version.sha256}>SHA-256 {c.version.sha256.slice(0, 16)}…</dd>
      </dl>
      {c.newer_version ? (
        <p className="qc-note"><AlertTriangle size={15} aria-hidden /> A newer version ({c.newer_version}) of this document exists. This quote is from version {c.version.number}.</p>
      ) : null}
      <div className="qc-actions">
        {isWorkspacePath(c.reader_path) && <Link className="qc-btn" to={c.reader_path}><FileText aria-hidden />Open full document</Link>}
        {onUse && <button type="button" className="qc-btn" onClick={onUse}><Plus aria-hidden />Use this document</button>}
      </div>
    </>
  );
}

function SideBody({ side }: { side: ComparisonSide }) {
  return (
    <>
      <blockquote className="qc-panel-quote" aria-label="Quoted passage">
        <Highlighted text={side.context} start={side.value_start} end={side.value_end} />
      </blockquote>
      <dl className="qc-facts">
        <dt>Document</dt><dd>{side.document_title}</dd>
        {side.version_number ? <><dt>Version</dt><dd>{side.version_number}{side.version_status ? ` · ${statusWord(side.version_status)}` : ''}</dd></> : null}
        {side.page ? <><dt>Page</dt><dd>{side.page}</dd></> : null}
        {side.clause ? <><dt>Clause</dt><dd>{side.clause}</dd></> : null}
      </dl>
      {side.stale && <p className="qc-note"><AlertTriangle size={15} aria-hidden /> A newer version of this document is now in use.</p>}
      {isWorkspacePath(side.reader_path) && (
        <div className="qc-actions"><Link className="qc-btn" to={side.reader_path}><FileText aria-hidden />Open full document</Link></div>
      )}
    </>
  );
}

function titleOf(c: PanelContent): string {
  return c.kind === 'citation' ? 'Source' : c.label;
}

export default function SidePanel() {
  const { content, close } = usePanel();
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => { if (content) heading.current?.focus(); }, [content]);
  if (!content) return null;
  return (
    <aside className="qc-panel" aria-label={titleOf(content)} role="complementary">
      <div className="qc-panel-head">
        <h2 ref={heading} tabIndex={-1}>{titleOf(content)}</h2>
        <button type="button" className="qc-icon-btn" onClick={close} aria-label="Close panel"><X aria-hidden /></button>
      </div>
      <div className="qc-panel-body">
        {content.kind === 'citation' ? <CitationBody key={content.id} id={content.id} onUse={content.onUse} /> : <SideBody side={content.side} />}
      </div>
    </aside>
  );
}
