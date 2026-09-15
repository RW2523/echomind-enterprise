import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { AnalysisCard, AnalysisLabel, EvidenceQuote, SentenceCheck, TagSpec } from '../types';
import { ICONS } from '../constants';
import { documentFileUrl } from '../services/backend';
import TagChip from './TagChip';
import {
  asCheck, copyToClipboard, flagTone, isSupported, primaryTagId, roleLabel, sourceLine, tagSpecFor, toneClasses,
} from '../utils/silentAssistant';
import { useHandraise } from '../hooks/useHandraise';

/**
 * Legacy v1 label styling — fallback for cards without tags/verdict.
 * Colour rules: strong colour ONLY for Contradicted (red), Violating (dark red), Risky (orange);
 * Supported = subtle green; Unverified / Relevant = neutral slate (no yellow, no warning glyph).
 */
export const LABEL_CONFIG: Record<AnalysisLabel, { bg: string; border: string; text: string; badge: string; icon: string }> = {
  Supported:         { bg: 'bg-emerald-500/[0.06]', border: 'border-emerald-500/25', text: 'text-emerald-300', badge: 'bg-emerald-500/15 text-emerald-300', icon: '✓' },
  Contradicted:      { bg: 'bg-rose-500/10',        border: 'border-rose-500/40',    text: 'text-rose-300',    badge: 'bg-rose-500/20 text-rose-300',       icon: '✗' },
  Unverified:        { bg: 'bg-white/[0.03]',       border: 'border-slate-500/40',   text: 'text-slate-400',   badge: 'bg-transparent text-slate-300',      icon: '?' },
  Violating:         { bg: 'bg-red-950/40',         border: 'border-red-700/60',     text: 'text-red-300',     badge: 'bg-red-900/40 text-red-200',         icon: '⊘' },
  'Risky Statement': { bg: 'bg-orange-500/10',      border: 'border-orange-500/40',  text: 'text-orange-300',  badge: 'bg-orange-500/20 text-orange-300',   icon: '⚡' },
  Relevant:          { bg: 'bg-white/[0.03]',       border: 'border-white/10',       text: 'text-cyan-300',    badge: 'bg-cyan-500/15 text-cyan-300',       icon: 'ℹ' },
};

/** Icon glyph for a check (by primary tag, falling back to legacy label). */
export function checkGlyph(check: SentenceCheck): string {
  const p = primaryTagId(check);
  switch (p) {
    case 'supported': return '✓';
    case 'contradicted': return '✗';
    case 'unverified': return '?';
    case 'violating': return '⊘';
    case 'risk': case 'disclosure-missing': return '⚡';
    case 'record-found': return '🗂';
    case 'personal-detail': return '👤';
    case 'contract-clause': return '§';
    case 'policy': return '📋';
    case 'related-case': return '📚';
    case 'action-item': return '☑';
    case 'commitment': return '🤝';
    case 'decision': return '⚖';
    case 'question': return '?';
    case 'reference': return 'ℹ';
    default: return LABEL_CONFIG[check.label]?.icon ?? 'ℹ';
  }
}

/**
 * Colour classes for a check card. Strong colour only when flagged (contradicted = red, violating = dark red,
 * risk / disclosure-missing = orange); supported = subtle green; everything else (unverified, records,
 * references, actions…) renders neutral — the small tag chips carry the per-tag colour.
 */
export function checkStyle(check: SentenceCheck, vocab?: TagSpec[]): { bg: string; border: string; text: string; bar: string; badge: string } {
  const flag = flagTone(check);
  if (flag) {
    const t = toneClasses(flag);
    return { bg: t.bg, border: t.border, text: t.text, bar: t.bar, badge: t.badge };
  }
  if (isSupported(check)) {
    const cfg = LABEL_CONFIG.Supported;
    return { bg: cfg.bg, border: cfg.border, text: cfg.text, bar: 'bg-emerald-500', badge: `${cfg.badge} border-emerald-500/30` };
  }
  const p = primaryTagId(check);
  const spec = p ? tagSpecFor(vocab, p, check.tags?.find((t) => t.tag === p)) : undefined;
  const t = toneClasses(spec?.tone);
  const neutralText = !spec || p === 'unverified' || p === 'question';
  return { bg: 'bg-white/[0.03]', border: 'border-white/10', text: neutralText ? 'text-slate-400' : t.text, bar: neutralText ? 'bg-slate-500' : t.bar, badge: t.badge };
}

/** Render `text` with `quote` wrapped in <mark> (case/whitespace-insensitive). Falls back to plain text. */
export function HighlightedText({ text, quote, markClass }: { text: string; quote?: string | null; markClass: string }) {
  const parts = useMemo(() => {
    if (!text) return null;
    const q = (quote ?? '').trim();
    if (!q) return null;
    const norm = (s: string) => s.toLowerCase().replace(/\s+/g, ' ');
    const hay = norm(text);
    const needle = norm(q);
    // Build a mapping from normalised index -> original index.
    const map: number[] = [];
    let ws = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (/\s/.test(ch)) { if (!ws) { map.push(i); ws = true; } }
      else { map.push(i); ws = false; }
    }
    let idx = hay.indexOf(needle);
    if (idx < 0) {
      // try the first ~8 words of the quote
      const short = needle.split(' ').slice(0, 8).join(' ');
      if (short.length >= 12) idx = hay.indexOf(short);
      if (idx < 0) return null;
      const s = map[idx], e = map[Math.min(idx + short.length, map.length - 1)] + 1;
      return { pre: text.slice(0, s), hit: text.slice(s, e), post: text.slice(e) };
    }
    const s = map[idx];
    const e = map[Math.min(idx + needle.length - 1, map.length - 1)] + 1;
    return { pre: text.slice(0, s), hit: text.slice(s, e), post: text.slice(e) };
  }, [text, quote]);
  if (!parts) return <>{text}</>;
  return (
    <>
      {parts.pre}
      <mark className={`rounded px-0.5 ${markClass}`}>{parts.hit}</mark>
      {parts.post}
    </>
  );
}

interface AnalysisCardModalProps {
  card: AnalysisCard | SentenceCheck;
  vocab?: TagSpec[];
  onClose: () => void;
}

const AnalysisCardModal: React.FC<AnalysisCardModalProps> = ({ card, vocab, onClose }) => {
  const check = useMemo(() => asCheck(card), [card]);
  const style = checkStyle(check, vocab);
  const glyph = checkGlyph(check);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const { speaking, speakCard, stop: stopSpeaking } = useHandraise();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const evidence: EvidenceQuote[] = check.evidence ?? [];
  const chunkText = (ev: EvidenceQuote): string | undefined =>
    (ev.chunk_id && check.source_chunks.find((c) => c.chunk_id === ev.chunk_id)?.text) || undefined;
  // Legacy source chunks not already shown as evidence.
  const extraChunks = check.source_chunks.filter((c) => !evidence.some((e) => e.chunk_id && e.chunk_id === c.chunk_id));

  const copy = async (key: string, text: string) => {
    if (await copyToClipboard(text)) { setCopied(key); setTimeout(() => setCopied(null), 1200); }
  };
  const copyAllEvidence = () => {
    const lines = evidence.map((e) => `“${e.quote}”${e.kind === 'rule' ? ` [rule ${e.rule_id ?? ''}]` : ''} — ${sourceLine(e.doc_title, e.page, e.section_path) || 'knowledge base'}`);
    copy('all', `${check.sentence_text}\n\n${lines.join('\n')}${check.explanation ? `\n\n${check.explanation}` : ''}`);
  };

  const confidence = Number.isFinite(check.confidence) ? check.confidence : 0;
  const tagIds = check.tags?.length ? check.tags : (primaryTagId(check) ? [{ tag: primaryTagId(check)! }] : []);

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onClick={(e) => { if (e.target === overlayRef.current) onClose(); }}
    >
      <div className="relative w-full max-w-2xl max-h-[90vh] flex flex-col rounded-2xl border border-white/10 bg-slate-900 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className={`shrink-0 flex items-center gap-3 px-5 py-4 border-b ${style.border} ${style.bg}`}>
          <span className={`text-xl font-bold ${style.text}`}>{glyph}</span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 flex-wrap">
              {tagIds.length ? tagIds.map((t) => <TagChip key={t.tag} tag={t} vocab={vocab} size="sm" showConfidence />) : (
                <span className={`text-xs font-semibold uppercase tracking-wider px-2.5 py-1 rounded-full border ${style.badge}`}>{check.label}</span>
              )}
              {check.role && (
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-300 bg-white/10 border border-white/10 rounded-full px-2 py-0.5">
                  {roleLabel(check.role)}
                </span>
              )}
              {check.kind && check.kind !== 'claim' && (
                <span className="text-[10px] text-slate-400">{roleLabel(check.kind)}</span>
              )}
              <span className="text-xs text-slate-400">{confidence.toFixed(0)}% confidence</span>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
            aria-label="Close"
          >
            <ICONS.Close className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 min-h-0 overflow-auto p-5 space-y-5">
          {/* Spoken statement */}
          <div>
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Spoken statement</div>
            <blockquote className={`rounded-xl border-l-4 ${style.border} pl-4 pr-3 py-3 ${style.bg} text-sm text-white/90 leading-relaxed italic`}>
              “{check.sentence_text || check.segment_text}”
            </blockquote>
          </div>

          {/* Evidence — first */}
          {evidence.length > 0 && (
            <div>
              <div className="flex items-center mb-2">
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Proof ({evidence.length})</div>
                <button
                  type="button"
                  onClick={copyAllEvidence}
                  className="ml-auto text-[11px] font-semibold text-slate-300 hover:text-white px-2 py-1 rounded-lg hover:bg-white/10"
                >
                  {copied === 'all' ? 'Copied' : 'Copy evidence'}
                </button>
              </div>
              <div className="space-y-3">
                {evidence.map((ev, i) => {
                  const isRule = ev.kind === 'rule';
                  const isTranscript = ev.kind === 'transcript';
                  const src = sourceLine(ev.doc_title, ev.page, ev.section_path);
                  const full = chunkText(ev);
                  const key = `ev-${i}`;
                  return (
                    <div key={key} className={`rounded-xl border p-4 ${isRule ? 'border-orange-500/30 bg-orange-500/5' : 'border-white/10 bg-white/5'}`}>
                      <div className="flex items-center gap-2 mb-2 flex-wrap">
                        {isRule ? (
                          <span className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded-md bg-orange-500/20 text-orange-300 border border-orange-500/40">Rule</span>
                        ) : isTranscript ? (
                          <span className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/40">Transcript</span>
                        ) : (
                          <ICONS.File className="w-4 h-4 text-cyan-400 shrink-0" />
                        )}
                        <span className={`text-xs font-medium truncate ${isRule ? 'text-orange-300' : 'text-cyan-400'}`}>
                          {isRule ? (ev.rule_id ? `rule: ${ev.rule_id}` : 'domain rule') : (src || 'Knowledge base')}
                        </span>
                        <div className="ml-auto flex items-center gap-1">
                          {ev.doc_id && !isRule && !isTranscript && (
                            <a
                              href={documentFileUrl(ev.doc_id, ev.page)}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-[11px] font-semibold text-cyan-300 hover:text-white px-2 py-1 rounded-lg hover:bg-white/10"
                            >
                              Open document{ev.page != null ? ` · p.${ev.page}` : ''}
                            </a>
                          )}
                          <button
                            type="button"
                            onClick={() => copy(key, `“${ev.quote}” — ${src || (isRule ? `rule ${ev.rule_id ?? ''}` : 'knowledge base')}`)}
                            className="text-[11px] font-semibold text-slate-300 hover:text-white px-2 py-1 rounded-lg hover:bg-white/10"
                          >
                            {copied === key ? 'Copied' : 'Copy'}
                          </button>
                        </div>
                      </div>
                      {full ? (
                        <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
                          <HighlightedText text={full} quote={ev.quote} markClass={toneClasses(isRule ? 'orange' : 'cyan').mark} />
                        </p>
                      ) : (
                        <blockquote className={`border-l-2 ${isRule ? 'border-orange-400/60' : 'border-cyan-400/60'} pl-3 text-sm text-white/90 leading-relaxed italic`}>
                          “{ev.quote}”
                        </blockquote>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Explanation */}
          {check.explanation && (
            <div>
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Why</div>
              <p className="text-sm text-slate-200 leading-relaxed">{check.explanation}</p>
            </div>
          )}

          {/* Confidence bar */}
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Confidence</span>
              <span className={style.text}>{confidence.toFixed(0)}%</span>
            </div>
            <div className="h-2 rounded-full bg-white/5 overflow-hidden">
              <div className={`h-full rounded-full transition-all ${style.bar}`} style={{ width: `${Math.max(0, Math.min(100, confidence))}%` }} />
            </div>
          </div>

          {/* Legacy / additional source chunks */}
          {extraChunks.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                {evidence.length ? 'Other sources' : 'Reference sources'} ({extraChunks.length})
              </div>
              <div className="space-y-3">
                {extraChunks.map((chunk, i) => (
                  <div key={chunk.chunk_id || i} className="rounded-xl border border-white/10 bg-white/5 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <ICONS.File className="w-4 h-4 text-cyan-400 shrink-0" />
                      <span className="text-xs font-medium text-cyan-400 truncate">{chunk.doc_title || 'Source'}</span>
                      {chunk.doc_id && (
                        <a href={documentFileUrl(chunk.doc_id)} target="_blank" rel="noopener noreferrer" className="ml-auto text-[11px] font-semibold text-cyan-300 hover:text-white px-2 py-1 rounded-lg hover:bg-white/10">
                          Open document
                        </a>
                      )}
                    </div>
                    <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">{chunk.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Searched docs / latency */}
          {(check.searched_docs?.length > 0 || check.latency_ms != null || check.llm_skipped) && (
            <div className="text-[10px] text-slate-500 flex flex-wrap gap-x-3 gap-y-1">
              {check.searched_docs?.length > 0 && <span>Searched: {check.searched_docs.slice(0, 6).join(', ')}{check.searched_docs.length > 6 ? ` +${check.searched_docs.length - 6}` : ''}</span>}
              {check.latency_ms != null && <span>{Math.round(check.latency_ms)} ms</span>}
              {check.llm_skipped && <span>rule/retrieval only</span>}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="shrink-0 px-5 py-3 border-t border-white/10 flex items-center gap-2">
          <button
            type="button"
            onClick={() => (speaking ? stopSpeaking() : speakCard(check))}
            className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-2 text-xs font-semibold transition-colors ${
              speaking ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-white/10 text-slate-300 hover:bg-white/15'
            }`}
            title="Read this check out loud (Handraise)"
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
            {speaking ? 'Stop reading' : 'Read out'}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="ml-auto rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 text-slate-300 hover:bg-white/15 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnalysisCardModal;
