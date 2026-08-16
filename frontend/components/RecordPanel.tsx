import React, { useMemo, useState } from 'react';
import type { RecordHit, Subject } from '../types';
import { documentFileUrl } from '../services/backend';
import { copyToClipboard, recordKindMeta, sourceLine, toneClasses } from '../utils/silentAssistant';

interface RecordPanelProps {
  records: RecordHit[];
  subjects?: Subject[];
  /** Highlight records pulled for this sentence */
  selectedSentenceId?: string | null;
  onSelectSentence?: (sentenceId: string | null) => void;
  emptyHint?: string;
}

const KIND_ORDER = ['customer_file', 'kyc', 'account', 'matter', 'contract', 'ticket', 'previous_call', 'related_case', 'product', 'policy', 'document'];

const RecordRow: React.FC<{ rec: RecordHit; subjectName?: string; highlighted: boolean; onClick?: () => void }> = ({ rec, subjectName, highlighted, onClick }) => {
  const [copied, setCopied] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const first = rec.quotes?.[0];
  const src = sourceLine(rec.doc_title, rec.page, rec.section_path);
  const canOpen = !!rec.doc_id && rec.kind !== 'previous_call';
  const score = typeof rec.score === 'number' ? Math.round(rec.score <= 1 ? rec.score * 100 : rec.score) : null;

  const onCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!first) return;
    if (await copyToClipboard(`${first.text}${src ? ` — ${src}` : ''}`)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }
  };

  return (
    <div
      onClick={onClick}
      className={`rounded-xl border p-3 transition-all ${highlighted ? 'border-sky-500/50 bg-sky-500/10 ring-1 ring-sky-500/30' : 'border-white/10 bg-white/5 hover:bg-white/[0.07]'} ${onClick ? 'cursor-pointer' : ''}`}
    >
      <div className="flex items-start gap-2">
        <div className="flex-1 min-w-0">
          <div className="text-xs font-semibold text-white leading-snug break-words">{rec.title}</div>
          <div className="text-[10px] text-slate-400 mt-0.5 flex flex-wrap items-center gap-x-1.5">
            {rec.doc_title && rec.doc_title !== rec.title && <span className="truncate max-w-[14rem]" title={rec.doc_title}>{rec.doc_title}</span>}
            {rec.page != null && <span>· p.{rec.page}</span>}
            {rec.section_path && <span className="truncate max-w-[10rem]" title={rec.section_path}>· {rec.section_path}</span>}
            {subjectName && <span className="text-violet-300">· {subjectName}</span>}
          </div>
        </div>
        <div className="shrink-0 flex items-center gap-1">
          {rec.match && (
            <span className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded-md border ${
              rec.match === 'exact' ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
              : rec.match === 'fuzzy' ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
              : 'bg-white/10 text-slate-300 border-white/15'
            }`} title={score != null ? `${score}% relevance` : rec.match}>
              {rec.match}
            </span>
          )}
        </div>
      </div>
      {first && (
        <blockquote className="mt-2 border-l-2 border-sky-400/60 pl-2.5 text-[11.5px] leading-relaxed text-white/85 italic">
          “{first.text}”
        </blockquote>
      )}
      {showAll && rec.quotes.slice(1).map((q, i) => (
        <blockquote key={i} className="mt-1.5 border-l-2 border-white/20 pl-2.5 text-[11px] leading-relaxed text-white/70 italic">
          “{q.text}”
        </blockquote>
      ))}
      <div className="mt-2 flex items-center gap-1.5 flex-wrap">
        {canOpen && (
          <a
            href={documentFileUrl(rec.doc_id!, rec.page)}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[10px] font-semibold bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/25 transition-colors"
          >
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
            Open{rec.page != null ? ` p.${rec.page}` : ''}
          </a>
        )}
        {rec.kind === 'previous_call' && rec.source_transcript_id && (
          <span className="text-[10px] text-slate-500" title={rec.source_transcript_id}>transcript {rec.source_transcript_id.slice(0, 8)}…</span>
        )}
        {first && (
          <button
            type="button"
            onClick={onCopy}
            className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[10px] font-semibold bg-white/10 text-slate-300 border border-white/10 hover:bg-white/15 transition-colors"
          >
            {copied ? 'Copied' : 'Copy quote'}
          </button>
        )}
        {rec.quotes.length > 1 && (
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); setShowAll((v) => !v); }}
            className="text-[10px] text-slate-400 hover:text-white ml-auto"
          >
            {showAll ? 'Less' : `+${rec.quotes.length - 1} more`}
          </button>
        )}
      </div>
    </div>
  );
};

/** Records grouped by kind (customer file / contract / ticket / previous call / matter / …). */
const RecordPanel: React.FC<RecordPanelProps> = ({ records, subjects, selectedSentenceId, onSelectSentence, emptyHint }) => {
  const groups = useMemo(() => {
    const m = new Map<string, RecordHit[]>();
    for (const r of records) {
      const k = r.kind || 'document';
      if (!m.has(k)) m.set(k, []);
      m.get(k)!.push(r);
    }
    const keys = [...m.keys()].sort((a, b) => {
      const ia = KIND_ORDER.indexOf(a), ib = KIND_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    return keys.map((k) => ({ kind: k, items: m.get(k)!.slice().sort((a, b) => (b.score ?? 0) - (a.score ?? 0)) }));
  }, [records]);

  const subjectName = (id?: string | null) => (id ? subjects?.find((s) => s.id === id)?.display_name : undefined);

  if (!records.length) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-6 gap-2">
        <div className="w-12 h-12 rounded-2xl border border-white/10 bg-white/5 flex items-center justify-center text-xl">🗂️</div>
        <div className="text-sm font-medium text-slate-400">No records yet</div>
        <p className="text-xs text-slate-500 max-w-[220px] leading-relaxed">
          {emptyHint ?? 'When a name, account, order or case number is spoken, matching records from your documents and previous calls appear here.'}
        </p>
      </div>
    );
  }

  return (
    <div className="p-2 space-y-3">
      {groups.map(({ kind, items }) => {
        const meta = recordKindMeta(kind);
        const tone = toneClasses(meta.tone);
        return (
          <section key={kind}>
            <div className="flex items-center gap-1.5 px-1 mb-1.5">
              <span className="text-sm leading-none">{meta.icon}</span>
              <span className={`text-[10px] font-bold uppercase tracking-widest ${tone.text}`}>{meta.label}</span>
              <span className="text-[10px] text-slate-500 bg-white/5 rounded-full px-1.5">{items.length}</span>
            </div>
            <div className="space-y-2">
              {items.map((rec) => (
                <RecordRow
                  key={rec.id}
                  rec={rec}
                  subjectName={subjectName(rec.subject_id)}
                  highlighted={!!selectedSentenceId && rec.sentence_id === selectedSentenceId}
                  onClick={onSelectSentence && rec.sentence_id ? () => onSelectSentence(rec.sentence_id === selectedSentenceId ? null : rec.sentence_id!) : undefined}
                />
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
};

export default RecordPanel;
