import React from 'react';
import type { CheckStatus, Role, SentenceCheck, TagSpec, TranscriptSegment, TranscriptSentence } from '../types';
import { LABEL_CONFIG, checkStyle } from './AnalysisCardModal';
import TagChip from './TagChip';
import ProofPopover from './ProofPopover';
import { legacyLabelToTag, roleLabel, statusGlyph } from '../utils/silentAssistant';

/** Small speaker chip; "me" role = cyan, "other" = violet, unknown = slate. */
export const RoleChip: React.FC<{ role?: Role | null; roles?: { me: Role; other: Role }; className?: string }> = ({ role, roles, className = '' }) => {
  if (!role) return null;
  const isMe = roles && role === roles.me;
  const isOther = roles && role === roles.other;
  const cls = isMe ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30' : isOther ? 'bg-violet-500/15 text-violet-300 border-violet-500/30' : 'bg-white/10 text-slate-300 border-white/15';
  return (
    <span className={`inline-flex items-center align-middle rounded-md border px-1.5 py-px text-[9px] font-bold uppercase tracking-widest leading-none mr-1.5 ${cls} ${className}`}>
      {roleLabel(role)}
    </span>
  );
};

interface SentenceSpanProps {
  sentence: TranscriptSentence;
  text: string;
  check?: SentenceCheck;
  status?: CheckStatus;
  vocab?: TagSpec[];
  selected: boolean;
  onSelect: () => void;
  showRole?: boolean;
  roles?: { me: Role; other: Role };
}

const SentenceSpan: React.FC<SentenceSpanProps> = ({ sentence, text, check, status, vocab, selected, onSelect, showRole, roles }) => {
  const st = check ? checkStyle(check, vocab) : null;
  const glyph = statusGlyph(check ? 'checked' : status);
  const first = check?.evidence?.[0];
  const clickable = !!check;

  const inner = (
    <span
      onClick={clickable ? (e) => { e.stopPropagation(); onSelect(); } : undefined}
      className={`rounded px-0.5 transition-colors ${
        st ? `${st.bg} border-b ${st.border} cursor-pointer hover:brightness-125` : 'text-white/90'
      } ${selected ? 'ring-1 ring-white/40 brightness-125' : ''}`}
      title={check ? `${check.tags?.map((t) => t.label ?? t.tag).join(', ') || check.label} · ${Math.round(check.confidence)}%` : glyph.title || undefined}
      data-sentence-id={sentence.sentence_id}
    >
      {showRole && <RoleChip role={sentence.role} roles={roles} />}
      {glyph.glyph && (
        <span className={`text-[10px] font-bold mr-1 align-middle ${glyph.cls}`} aria-label={glyph.title}>{glyph.glyph}</span>
      )}
      <span className={st ? 'text-white' : ''}>{text}</span>
      {check && check.tags?.length > 0 && (
        <span className="inline-flex items-center gap-1 ml-1.5 align-middle">
          {check.tags.slice(0, 2).map((t) => <TagChip key={t.tag} tag={t} vocab={vocab} />)}
          {check.tags.length > 2 && <span className="text-[9px] text-slate-400">+{check.tags.length - 2}</span>}
        </span>
      )}
      {check && !check.tags?.length && (
        <span className="inline-flex items-center ml-1.5 align-middle">
          <TagChip tag={{ tag: legacyLabelToTag(check.label) ?? 'reference', label: check.label }} vocab={vocab} />
        </span>
      )}
    </span>
  );

  if (first) {
    return (
      <ProofPopover evidence={first} note={check?.explanation} trigger="hover">
        {inner}
      </ProofPopover>
    );
  }
  return inner;
};

interface SegmentLineProps {
  segment: TranscriptSegment;
  checks: Record<string, SentenceCheck>;
  sentenceStatus: Record<string, CheckStatus>;
  vocab?: TagSpec[];
  roles?: { me: Role; other: Role };
  selectedSentenceId: string | null;
  onSelectSentence: (id: string, check?: SentenceCheck) => void;
  /** Legacy (v1) segment-level selection */
  isSelected: boolean;
  onSelectSegment: () => void;
}

/**
 * One transcript paragraph. With v2 `sentences[]` each sentence is its own span
 * (role chip, status glyph, tag chips, proof popover); otherwise falls back to the
 * v1 segment-level highlight.
 */
export const TranscriptSegmentLine: React.FC<SegmentLineProps> = ({
  segment, checks, sentenceStatus, vocab, roles, selectedSentenceId, onSelectSentence, isSelected, onSelectSegment,
}) => {
  const sentences = segment.sentences;

  if (sentences && sentences.length > 0) {
    const text = segment.text ?? '';
    const parts: React.ReactNode[] = [];
    let cursor = 0;
    let lastRole: Role | null | undefined = undefined;
    sentences.forEach((s, i) => {
      const cs = s.char_start, ce = s.char_end;
      const validOffsets = Number.isFinite(cs) && Number.isFinite(ce) && cs >= 0 && ce > cs && ce <= text.length && cs >= cursor;
      let sText = s.text;
      if (validOffsets) {
        if (cs > cursor) parts.push(<span key={`gap-${i}`} className="text-white/90">{text.slice(cursor, cs)}</span>);
        sText = text.slice(cs, ce);
        cursor = ce;
      } else if (i > 0) {
        parts.push(<span key={`sp-${i}`}> </span>);
      }
      const check = checks[s.sentence_id];
      const role = s.role ?? segment.role;
      const showRole = role != null && role !== lastRole;
      lastRole = role;
      parts.push(
        <SentenceSpan
          key={s.sentence_id}
          sentence={{ ...s, role }}
          text={sText}
          check={check}
          status={sentenceStatus[s.sentence_id]}
          vocab={vocab}
          roles={roles}
          showRole={showRole}
          selected={selectedSentenceId === s.sentence_id}
          onSelect={() => onSelectSentence(s.sentence_id, check)}
        />
      );
    });
    if (cursor < text.length) parts.push(<span key="tail" className="text-white/90">{text.slice(cursor)}</span>);
    return <p className="leading-loose">{parts}</p>;
  }

  // ── v1 fallback: whole-segment highlight ──
  const segCheck = checks[segment.paragraph_id];
  const cfg = segment.label ? LABEL_CONFIG[segment.label] : segCheck ? LABEL_CONFIG[segCheck.label] : null;
  const st = segCheck ? checkStyle(segCheck, vocab) : null;
  const status = sentenceStatus[segment.paragraph_id];
  const glyph = statusGlyph(segCheck ? 'checked' : status);

  if (!cfg && !st) {
    return (
      <p className="text-white/90">
        <RoleChip role={segment.role} roles={roles} />
        {glyph.glyph && <span className={`text-[10px] font-bold mr-1 ${glyph.cls}`}>{glyph.glyph}</span>}
        {segment.text}
      </p>
    );
  }
  const border = st?.border ?? cfg!.border;
  const bg = st?.bg ?? cfg!.bg;
  const textCls = st?.text ?? cfg!.text;
  const icon = cfg?.icon ?? '•';
  const body = (
    <div
      onClick={() => { onSelectSegment(); if (segCheck) onSelectSentence(segment.paragraph_id, segCheck); }}
      className={`cursor-pointer transition-all duration-200 rounded px-1 py-0.5 border-l-2 ${border} ${
        isSelected || selectedSentenceId === segment.paragraph_id ? `${bg} brightness-125` : `${bg} opacity-90 hover:opacity-100`
      }`}
      title={`${segCheck?.tags?.map((t) => t.label ?? t.tag).join(', ') || segment.label} (${(segCheck?.confidence ?? segment.confidence ?? 0).toFixed(0)}%)`}
    >
      <RoleChip role={segment.role} roles={roles} />
      <span className={`text-[10px] font-bold mr-1.5 ${textCls}`}>{icon}</span>
      <span className="text-white/90">{segment.text}</span>
      {segCheck?.tags?.length ? (
        <span className="inline-flex items-center gap-1 ml-1.5 align-middle">
          {segCheck.tags.slice(0, 2).map((t) => <TagChip key={t.tag} tag={t} vocab={vocab} />)}
        </span>
      ) : null}
    </div>
  );
  const first = segCheck?.evidence?.[0];
  return first ? <ProofPopover evidence={first} note={segCheck?.explanation} trigger="hover" block>{body}</ProofPopover> : body;
};

export default TranscriptSegmentLine;
